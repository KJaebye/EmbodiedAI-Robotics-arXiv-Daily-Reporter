# AssemMate: Graph-Based LLM for Robotic Assembly Assistance 

**Title (ZH)**: AssemMate：基于图的大型语言模型在机器人装配辅助中的应用 

**Authors**: Qi Zheng, Chaoran Zhang, Zijian Liang, EnTe Lin, Shubo Cui, Qinghongbing Xie, Zhaobo Xu, Long Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2509.11617)  

**Abstract**: Large Language Model (LLM)-based robotic assembly assistance has gained significant research attention. It requires the injection of domain-specific knowledge to guide the assembly process through natural language interaction with humans. Despite some progress, existing methods represent knowledge in the form of natural language text. Due to the long context and redundant content, they struggle to meet the robots' requirements for real-time and precise reasoning. In order to bridge this gap, we present AssemMate, which utilizes the graph\textemdash a concise and accurate form of knowledge representation\textemdash as input. This graph-based LLM enables knowledge graph question answering (KGQA), supporting human-robot interaction and assembly task planning for specific products. Beyond interactive QA, AssemMate also supports sensing stacked scenes and executing grasping to assist with assembly. Specifically, a self-supervised Graph Convolutional Network (GCN) encodes knowledge graph entities and relations into a latent space and aligns them with LLM's representation, enabling the LLM to understand graph information. In addition, a vision-enhanced strategy is employed to address stacked scenes in grasping. Through training and evaluation, AssemMate outperforms existing methods, achieving 6.4\% higher accuracy, 3 times faster inference, and 28 times shorter context length, while demonstrating strong generalization ability on random graphs. And our approach further demonstrates superiority through robotic grasping experiments in both simulated and real-world settings. More details can be found on the project page: this https URL 

**Abstract (ZH)**: 基于大型语言模型（LLM）的机器人装配辅助：利用图表示的知识驱动交互与装配任务规划 

---
# Large Foundation Models for Trajectory Prediction in Autonomous Driving: A Comprehensive Survey 

**Title (ZH)**: 大型基础模型在自主驾驶中的轨迹预测：一篇综述 

**Authors**: Wei Dai, Shengen Wu, Wei Wu, Zhenhao Wang, Sisuo Lyu, Haicheng Liao, Limin Yu, Weiping Ding, Runwei Guan, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2509.10570)  

**Abstract**: Trajectory prediction serves as a critical functionality in autonomous driving, enabling the anticipation of future motion paths for traffic participants such as vehicles and pedestrians, which is essential for driving safety. Although conventional deep learning methods have improved accuracy, they remain hindered by inherent limitations, including lack of interpretability, heavy reliance on large-scale annotated data, and weak generalization in long-tail scenarios. The rise of Large Foundation Models (LFMs) is transforming the research paradigm of trajectory prediction. This survey offers a systematic review of recent advances in LFMs, particularly Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) for trajectory prediction. By integrating linguistic and scene semantics, LFMs facilitate interpretable contextual reasoning, significantly enhancing prediction safety and generalization in complex environments. The article highlights three core methodologies: trajectory-language mapping, multimodal fusion, and constraint-based reasoning. It covers prediction tasks for both vehicles and pedestrians, evaluation metrics, and dataset analyses. Key challenges such as computational latency, data scarcity, and real-world robustness are discussed, along with future research directions including low-latency inference, causality-aware modeling, and motion foundation models. 

**Abstract (ZH)**: Large Foundation Models在轨迹预测中的最新进展：面向自主驾驶的应用 

---
# Advancing Medical Artificial Intelligence Using a Century of Cases 

**Title (ZH)**: 利用百年病例推动医疗人工智能发展 

**Authors**: Thomas A. Buckley, Riccardo Conci, Peter G. Brodeur, Jason Gusdorf, Sourik Beltrán, Bita Behrouzi, Byron Crowe, Jacob Dockterman, Muzzammil Muhammad, Sarah Ohnigian, Andrew Sanchez, James A. Diao, Aashna P. Shah, Daniel Restrepo, Eric S. Rosenberg, Andrew S. Lea, Marinka Zitnik, Scott H. Podolsky, Zahir Kanjee, Raja-Elie E. Abdulnour, Jacob M. Koshy, Adam Rodman, Arjun K. Manrai  

**Link**: [PDF](https://arxiv.org/pdf/2509.12194)  

**Abstract**: BACKGROUND: For over a century, the New England Journal of Medicine Clinicopathological Conferences (CPCs) have tested the reasoning of expert physicians and, recently, artificial intelligence (AI). However, prior AI evaluations have focused on final diagnoses without addressing the multifaceted reasoning and presentation skills required of expert discussants.
METHODS: Using 7102 CPCs (1923-2025) and 1021 Image Challenges (2006-2025), we conducted extensive physician annotation and automated processing to create CPC-Bench, a physician-validated benchmark spanning 10 text-based and multimodal tasks, against which we evaluated leading large language models (LLMs). Then, we developed "Dr. CaBot," an AI discussant designed to produce written and slide-based video presentations using only the case presentation, modeling the role of the human expert in these cases.
RESULTS: When challenged with 377 contemporary CPCs, o3 (OpenAI) ranked the final diagnosis first in 60% of cases and within the top ten in 84% of cases, outperforming a 20-physician baseline; next-test selection accuracy reached 98%. Event-level physician annotations quantified AI diagnostic accuracy per unit of information. Performance was lower on literature search and image tasks; o3 and Gemini 2.5 Pro (Google) achieved 67% accuracy on image challenges. In blinded comparisons of CaBot vs. human expert-generated text, physicians misclassified the source of the differential in 46 of 62 (74%) of trials, and scored CaBot more favorably across quality dimensions. To promote research, we are releasing CaBot and CPC-Bench.
CONCLUSIONS: LLMs exceed physician performance on complex text-based differential diagnosis and convincingly emulate expert medical presentations, but image interpretation and literature retrieval remain weaker. CPC-Bench and CaBot may enable transparent and continued tracking of progress in medical AI. 

**Abstract (ZH)**: 背景：百年来，New England Journal of Medicine的临床病理讨论会（CPCs）一直测试着专家医师的推理能力和近期的人工智能（AI）的能力。然而，之前的AI评估大多集中在最终诊断上，而没有涉及专家讨论者所必需的多方面推理和 Presentation 技能。
方法：利用1923年至2025年的7102次CPCs和2006年至2025年的1021个影像挑战，我们进行了广泛的医师注释和自动化处理，构建了CPC-Bench，这是一个涵盖10项文本和多模态任务的医师验证基准，用于评估领先的大语言模型（LLMs）。然后，我们开发了“Dr. CaBot”，一个仅基于病例展示即可生成书面和幻灯片视频演示的AI讨论者，模拟这些病例中的人类专家角色。
结果：面对377个现代CPCs的挑战，o3（OpenAI）在60%的情况下将最终诊断排名第一，在84%的情况下位居前十，超越了20名医师的基准；二次测试选择准确率达到了98%。事件级别的医师注释量化了AI诊断的准确性，每单位信息的准确度。在文献搜索和影像任务方面，性能较低；o3和Gemini 2.5 Pro（Google）在影像挑战方面的准确率为67%。在盲测中，CaBot与人类专家生成的文字比较，医师在62次试验中有46次（74%）错误地分类了差异来源，并且在质量维度上对CaBot的评价更为积极。为了促进研究，我们正在释放CaBot和CPC-Bench。
结论：大语言模型在复杂文本基于的鉴别诊断方面超过了医师的表现，并且能令人信服地模拟专家医疗展示，但在影像解释和文献检索方面仍较弱。CPC-Bench和CaBot有可能促进透明并持续跟踪医学AI的进步。 

---
# JustEva: A Toolkit to Evaluate LLM Fairness in Legal Knowledge Inference 

**Title (ZH)**: JustEva: 一个评估法律知识推理中LLM公平性的工具包 

**Authors**: Zongyue Xue, Siyuan Zheng, Shaochun Wang, Yiran Hu, Shenran Wang, Yuxin Yao, Haitao Li, Qingyao Ai, Yiqun Liu, Yun Liu, Weixing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.12104)  

**Abstract**: The integration of Large Language Models (LLMs) into legal practice raises pressing concerns about judicial fairness, particularly due to the nature of their "black-box" processes. This study introduces JustEva, a comprehensive, open-source evaluation toolkit designed to measure LLM fairness in legal tasks. JustEva features several advantages: (1) a structured label system covering 65 extra-legal factors; (2) three core fairness metrics - inconsistency, bias, and imbalanced inaccuracy; (3) robust statistical inference methods; and (4) informative visualizations. The toolkit supports two types of experiments, enabling a complete evaluation workflow: (1) generating structured outputs from LLMs using a provided dataset, and (2) conducting statistical analysis and inference on LLMs' outputs through regression and other statistical methods. Empirical application of JustEva reveals significant fairness deficiencies in current LLMs, highlighting the lack of fair and trustworthy LLM legal tools. JustEva offers a convenient tool and methodological foundation for evaluating and improving algorithmic fairness in the legal domain. 

**Abstract (ZH)**: Large Language Models (LLMs)在法律实践中的集成引发了关于司法公平性的紧迫关切，特别是由于它们“黑箱”过程的性质。本研究介绍了JustEva，一个全面的开源评估工具包，旨在衡量LLM在法律任务中的公平性。JustEva具有以下优点：（1）涵盖65个非法律因素的结构化标签系统；（2）三个核心公平性指标——不一致性、偏差和不平衡不准确；（3）稳健的统计推断方法；以及（4）信息性可视化。该工具包支持两种类型的实验，能够完成完整的评估工作流程：（1）使用提供的数据集从LLM生成结构化输出，（2）通过回归和其他统计方法对LLM输出进行统计分析和推断。JustEva的实际应用揭示了当前LLM存在的显著公平性缺陷，强调了缺乏公平可靠的LLM法律工具的问题。JustEva为评估和改善法律领域的算法公平性提供了便捷的工具和方法论基础。 

---
# When Safe Unimodal Inputs Collide: Optimizing Reasoning Chains for Cross-Modal Safety in Multimodal Large Language Models 

**Title (ZH)**: 当安全的单模态输入冲突时：优化跨模态安全性的多模态大型语言模型中的推理链 

**Authors**: Wei Cai, Shujuan Liu, Jian Zhao, Ziyan Shi, Yusheng Zhao, Yuchen Yuan, Tianle Zhang, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.12060)  

**Abstract**: Multimodal Large Language Models (MLLMs) are susceptible to the implicit reasoning risk, wherein innocuous unimodal inputs synergistically assemble into risky multimodal data that produce harmful outputs. We attribute this vulnerability to the difficulty of MLLMs maintaining safety alignment through long-chain reasoning. To address this issue, we introduce Safe-Semantics-but-Unsafe-Interpretation (SSUI), the first dataset featuring interpretable reasoning paths tailored for such a cross-modal challenge. A novel training framework, Safety-aware Reasoning Path Optimization (SRPO), is also designed based on the SSUI dataset to align the MLLM's internal reasoning process with human safety values. Experimental results show that our SRPO-trained models achieve state-of-the-art results on key safety benchmarks, including the proposed Reasoning Path Benchmark (RSBench), significantly outperforming both open-source and top-tier commercial MLLMs. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）容易受到隐式推理风险的威胁，其中无害的单模态输入在多模态数据中协同组装，产生有害的输出。我们将其脆弱性归因于MLLMs在长时间链推理中保持安全对齐的困难。为解决这一问题，我们引入了安全语义但有害解释（SSUI）数据集，这是首个专门为这种跨模态挑战设计的可解释推理路径数据集。基于SSUI数据集，我们还设计了一种新的训练框架——安全意识推理路径优化（SRPO），以使MLLM的内部推理过程与人类的安全价值观保持一致。实验结果表明，我们的SRPO训练模型在关键的安全基准测试中取得了最先进的成果，包括提出的推理路径基准（RSBench），显著优于开源和顶级商用MLLMs。 

---
# Adapting and Evaluating Multimodal Large Language Models for Adolescent Idiopathic Scoliosis Self-Management: A Divide and Conquer Framework 

**Title (ZH)**: 适用于青少年特发性脊柱侧弯自我管理的多模态大型语言模型的适应与评估：一种分而治之框架 

**Authors**: Zhaolong Wu, Pu Luo, Jason Pui Yin Cheung, Teng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11645)  

**Abstract**: This study presents the first comprehensive evaluation of Multimodal Large Language Models (MLLMs) for Adolescent Idiopathic Scoliosis (AIS) self-management. We constructed a database of approximately 3,000 anteroposterior X-rays with diagnostic texts and evaluated five MLLMs through a `Divide and Conquer' framework consisting of a visual question-answering task, a domain knowledge assessment task, and a patient education counseling assessment task. Our investigation revealed limitations of MLLMs' ability in interpreting complex spinal radiographs and comprehending AIS care knowledge. To address these, we pioneered enhancing MLLMs with spinal keypoint prompting and compiled an AIS knowledge base for retrieval augmented generation (RAG), respectively. Results showed varying effectiveness of visual prompting across different architectures, while RAG substantially improved models' performances on the knowledge assessment task. Our findings indicate current MLLMs are far from capable in realizing personalized assistant in AIS care. The greatest challenge lies in their abilities to obtain accurate detections of spinal deformity locations (best accuracy: 0.55) and directions (best accuracy: 0.13). 

**Abstract (ZH)**: 本研究首次全面评估了多模态大型语言模型（MLLMs）在青少年特发性脊柱侧弯（AIS）自我管理中的应用。我们构建了一个包含约3,000张前后位X光片及其诊断文本的数据库，并通过一个由视觉问答任务、领域知识评估任务和患者教育咨询评估任务组成的“分而治之”框架评估了五种MLLMs。我们的研究揭示了MLLMs在解读复杂脊柱X光片和理解AIS护理知识方面的局限性。为解决上述问题，我们分别提出了脊柱关键点提示增强MLLMs和为检索增强生成（RAG）编纂AIS知识库的方法。结果显示，视觉提示在不同架构下的效果各异，而RAG显著提高了模型在知识评估任务中的表现。研究结果表明，当前的MLLMs在实现AIS护理中的个性化助手方面还远不成熟。最大的挑战在于他们准确检测脊柱畸形位置（最佳准确率：0.55）和方向（最佳准确率：0.13）的能力有限。 

---
# A Survey of Reasoning and Agentic Systems in Time Series with Large Language Models 

**Title (ZH)**: 大规模语言模型在时间序列中的推理与代理系统综述 

**Authors**: Ching Chang, Yidan Shi, Defu Cao, Wei Yang, Jeehyun Hwang, Haixin Wang, Jiacheng Pang, Wei Wang, Yan Liu, Wen-Chih Peng, Tien-Fu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.11575)  

**Abstract**: Time series reasoning treats time as a first-class axis and incorporates intermediate evidence directly into the answer. This survey defines the problem and organizes the literature by reasoning topology with three families: direct reasoning in one step, linear chain reasoning with explicit intermediates, and branch-structured reasoning that explores, revises, and aggregates. The topology is crossed with the main objectives of the field, including traditional time series analysis, explanation and understanding, causal inference and decision making, and time series generation, while a compact tag set spans these axes and captures decomposition and verification, ensembling, tool use, knowledge access, multimodality, agent loops, and LLM alignment regimes. Methods and systems are reviewed across domains, showing what each topology enables and where it breaks down in faithfulness or robustness, along with curated datasets, benchmarks, and resources that support study and deployment (this https URL). Evaluation practices that keep evidence visible and temporally aligned are highlighted, and guidance is distilled on matching topology to uncertainty, grounding with observable artifacts, planning for shift and streaming, and treating cost and latency as design budgets. We emphasize that reasoning structures must balance capacity for grounding and self-correction against computational cost and reproducibility, while future progress will likely depend on benchmarks that tie reasoning quality to utility and on closed-loop testbeds that trade off cost and risk under shift-aware, streaming, and long-horizon settings. Taken together, these directions mark a shift from narrow accuracy toward reliability at scale, enabling systems that not only analyze but also understand, explain, and act on dynamic worlds with traceable evidence and credible outcomes. 

**Abstract (ZH)**: 时间序列推理将时间视为一级维度，并直接将中间证据纳入答案中。本文综述定义了该领域的问题，并按照推理拓扑结构组织文献，分为三大类：一步直接推理、具有明确中间件的线性链推理以及探索、修订和聚合的分支结构推理。该拓扑结构与领域的主要目标交叉，包括传统的时序分析、解释与理解、因果推断与决策、时序生成，同时采用紧凑的标签集覆盖这些轴，捕捉分解与验证、集成、工具使用、知识访问、多模态、代理循环以及大语言模型对齐模式。本文综述了跨领域的技术和系统，展示了每种拓扑结构的优势以及其在忠实度或鲁棒性方面的局限性，并提供了精选的数据集、基准测试和资源以支持研究与部署。文中突出了保持证据可见性和时序对齐的评估实践，并提炼出匹配拓扑结构与不确定性、基于可观测特征进行定位、计划迁移与流式处理、以及将成本与延迟视为设计预算的指导原则。我们强调，推理结构必须在接地能力和自我纠正能力与计算成本与可重现性之间取得平衡，未来进展可能依赖于将推理质量与实用性连接在一起的基准测试和在迁移感知、流式处理和长时范围设置下权衡成本与风险的闭环试验平台。总体而言，这些方向标志着从狭隘的准确性向大规模可靠性转变，使系统不仅能进行分析，还能理解和解释具有可追溯证据和可信结果的动态世界。 

---
# MedicalOS: An LLM Agent based Operating System for Digital Healthcare 

**Title (ZH)**: MedicalOS: 一门基于大语言模型代理的数字健康操作系统 

**Authors**: Jared Zhu, Junde Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11507)  

**Abstract**: Decades' advances in digital health technologies, such as electronic health records, have largely streamlined routine clinical processes. Yet, most these systems are still hard to learn and use: Clinicians often face the burden of managing multiple tools, repeating manual actions for each patient, navigating complicated UI trees to locate functions, and spending significant time on administration instead of caring for patients. The recent rise of large language model (LLM) based agents demonstrates exceptional capability in coding and computer operation, revealing the potential for humans to interact with operating systems and software not by direct manipulation, but by instructing agents through natural language. This shift highlights the need for an abstraction layer, an agent-computer interface, that translates human language into machine-executable commands. In digital healthcare, however, requires a more domain-specific abstractions that strictly follow trusted clinical guidelines and procedural standards to ensure safety, transparency, and compliance. To address this need, we present \textbf{MedicalOS}, a unified agent-based operational system designed as such a domain-specific abstract layer for healthcare. It translates human instructions into pre-defined digital healthcare commands, such as patient inquiry, history retrieval, exam management, report generation, referrals, treatment planning, that we wrapped as off-the-shelf tools using machine languages (e.g., Python, APIs, MCP, Linux). We empirically validate MedicalOS on 214 patient cases across 22 specialties, demonstrating high diagnostic accuracy and confidence, clinically sound examination requests, and consistent generation of structured reports and medication recommendations. These results highlight MedicalOS as a trustworthy and scalable foundation for advancing workflow automation in clinical practice. 

**Abstract (ZH)**: 几十年来数字健康技术的进展，如电子健康记录，大大简化了常规临床流程。然而，大多数这些系统仍然难以学习和使用：临床医生往往需要管理多个工具，为每位患者重复手动操作，通过复杂的用户界面导航来定位功能，并花费大量时间在行政事务上，而不是照顾病人。近期基于大型语言模型（LLM）的智能代理展示了在编码和计算机操作方面的卓越能力，揭示了人类可以通过自然语言指令智能代理与操作系统和软件交互的潜力。这一转变突显了需要一个抽象层，即代理-计算机接口，将人类语言转换为可执行的机器命令。在数字医疗保健领域，需要更具体的领域抽象，严格遵循可信赖的临床指南和程序标准，以确保安全、透明和合规。为应对这一需求，我们提出了**MedicalOS**，这是一个基于代理的统一操作系统，旨在作为这样一个特定领域的抽象层用于医疗保健。它将人类指令转换为预定义的数字医疗保健命令，如患者查询、病史检索、检查管理、报告生成、转诊、治疗规划等，我们使用机器语言（如Python、APIs、MCP、Linux）将这些命令打包为现成的工具。我们在22个专科的214个患者案例上实证验证了MedicalOS，结果显示高度准确的诊断和高置信度、临床合理的检查请求，以及一致生成的结构化报告和药物建议。这些结果突显了MedicalOS作为在临床实践中推进工作流自动化的信任和可扩展基础的重要性。 

---
# Securing AI Agents: Implementing Role-Based Access Control for Industrial Applications 

**Title (ZH)**: 基于角色的访问控制在工业应用中保障AI代理的安全 

**Authors**: Aadil Gani Ganie  

**Link**: [PDF](https://arxiv.org/pdf/2509.11431)  

**Abstract**: The emergence of Large Language Models (LLMs) has significantly advanced solutions across various domains, from political science to software development. However, these models are constrained by their training data, which is static and limited to information available up to a specific date. Additionally, their generalized nature often necessitates fine-tuning -- whether for classification or instructional purposes -- to effectively perform specific downstream tasks. AI agents, leveraging LLMs as their core, mitigate some of these limitations by accessing external tools and real-time data, enabling applications such as live weather reporting and data analysis. In industrial settings, AI agents are transforming operations by enhancing decision-making, predictive maintenance, and process optimization. For example, in manufacturing, AI agents enable near-autonomous systems that boost productivity and support real-time decision-making. Despite these advancements, AI agents remain vulnerable to security threats, including prompt injection attacks, which pose significant risks to their integrity and reliability. To address these challenges, this paper proposes a framework for integrating Role-Based Access Control (RBAC) into AI agents, providing a robust security guardrail. This framework aims to support the effective and scalable deployment of AI agents, with a focus on on-premises implementations. 

**Abstract (ZH)**: 大型语言模型（LLMs）的涌现显著推动了各个领域解决方案的发展，从政治科学到软件开发。然而，这些模型受限于其静态训练数据，这些数据仅限于某个特定日期之前的信息。此外，它们的通用性质通常需要微调——无论是为了分类还是指导性目的——以便有效地执行具体的下游任务。借助大型语言模型作为核心的AI代理通过访问外部工具和实时数据来部分缓解这些限制，这使它们能够用于诸如实时天气报告和数据分析等应用。在工业环境中，AI代理通过增强决策、预测维护和过程优化来转变运营。例如，在制造业中，AI代理使近自主系统得以实现，从而提高生产效率并支持实时决策。尽管取得了这些进展，AI代理仍然容易受到安全威胁的影响，包括提示注入攻击，这对其完整性和可靠性构成了重大风险。为应对这些挑战，本文提出了一种框架，将基于角色的访问控制（RBAC）集成到AI代理中，从而提供一个稳健的安全护栏。该框架旨在支持AI代理的有效和可扩展部署，特别注重本地实施。 

---
# MAPGD: Multi-Agent Prompt Gradient Descent for Collaborative Prompt Optimization 

**Title (ZH)**: MAPGD: 多代理提示梯度下降协作提示优化 

**Authors**: Yichen Han, Bojun Liu, Zhengpeng zhou, Guanyu Liu, Zeng Zhang, Yang Yang, Wenli Wang, Isaac N Shi, Yunyan, Lewei He, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2509.11361)  

**Abstract**: Prompt engineering is crucial for leveraging large language models (LLMs), but existing methods often rely on a single optimization trajectory, limiting adaptability and efficiency while suffering from narrow perspectives, gradient conflicts, and high computational cost. We propose MAPGD (Multi-Agent Prompt Gradient Descent), a framework integrating multi-agent collaboration with gradient-based optimization. MAPGD features specialized agents for task clarity, example selection, format design, and stylistic refinement; semantic gradient coordination to resolve conflicts; bandit-based candidate selection for efficient exploration-exploitation; and theoretical convergence guarantees. Experiments on classification, generation, and reasoning tasks show MAPGD outperforms single-agent and random baselines in accuracy and efficiency. Ablations confirm the benefits of gradient fusion, agent specialization, and conflict resolution, providing a unified, gradient-inspired multi-agent approach to robust and interpretable prompt optimization. 

**Abstract (ZH)**: 多代理提示梯度下降（MAPGD）：一种结合多代理协作与梯度优化的提示工程框架 

---
# Prompts to Proxies: Emulating Human Preferences via a Compact LLM Ensemble 

**Title (ZH)**: 从提示到代理：通过紧凑的LLMensemble模拟人类偏好 

**Authors**: Bingchen Wang, Zi-Yu Khoo, Bryan Kian Hsiang Low  

**Link**: [PDF](https://arxiv.org/pdf/2509.11311)  

**Abstract**: Large language models (LLMs) have demonstrated promise in emulating human-like responses across a wide range of tasks. In this paper, we propose a novel alignment framework that treats LLMs as agent proxies for human survey respondents, affording a cost-effective and steerable solution to two pressing challenges in the social sciences: the rising cost of survey deployment and the growing demographic imbalance in survey response data. Drawing inspiration from the theory of revealed preference, we formulate alignment as a two-stage problem: constructing diverse agent personas called endowments that simulate plausible respondent profiles, and selecting a representative subset to approximate a ground-truth population based on observed data. To implement the paradigm, we introduce P2P, a system that steers LLM agents toward representative behavioral patterns using structured prompt engineering, entropy-based sampling, and regression-based selection. Unlike personalization-heavy approaches, our alignment approach is demographic-agnostic and relies only on aggregate survey results, offering better generalizability and parsimony. Beyond improving data efficiency in social science research, our framework offers a testbed for studying the operationalization of pluralistic alignment. We demonstrate the efficacy of our approach on real-world opinion survey datasets, showing that our aligned agent populations can reproduce aggregate response patterns with high fidelity and exhibit substantial response diversity, even without demographic conditioning. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种任务上展示了模拟人类响应的潜力。本文提出了一种新颖的对齐框架，将LLMs视为人类调查受访者的人工智能代理，提供了一种成本效益高且可操控的解决方案，以应对社会科学中的两大挑战：调查部署成本上升和调查回应数据中的日益严重的种族群体失衡。借鉴揭示偏好的理论，我们将对齐问题转化为一个两阶段问题：构建多样化的代理人物个性（称为禀赋），模拟可信的受访者档案，以及根据观测数据选择代表性子集以逼近真实人口。为了实施这一范式，我们引入了P2P系统，通过结构化提示工程、基于熵的采样和基于回归的选择，引导LLM代理朝着代表性行为模式发展。与以个性化为主的方法不同，我们的对齐方法不依赖于人口统计信息，而是仅依赖于汇总的调查结果，从而提供更好的泛化能力和简洁性。除了在社会科学研究中提高数据效率外，我们的框架还为研究多元对齐的操作化提供了试验场。我们使用实际意见调查数据集验证了该方法的有效性，展示出我们的对齐代理群体能够高保真地再现总体响应模式，并且即使没有人口统计条件也能表现出显著的响应多样性。 

---
# Difficulty-Aware Agent Orchestration in LLM-Powered Workflows 

**Title (ZH)**: 基于LLM的强大工作流中具有难度感知的代理 orchestrator 

**Authors**: Jinwei Su, Yinghui Xia, Qizhen Lan, Xinyuan Song, Yang Jingsong, Lewei He, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2509.11079)  

**Abstract**: Large Language Model (LLM)-based agentic systems have shown strong capabilities across various tasks. However, ex- isting multi-agent frameworks often rely on static or task- level workflows, which either over-process simple queries or underperform on complex ones, while also neglecting the efficiency-performance trade-offs across heterogeneous LLMs. To address these limitations, we propose Difficulty- Aware Agentic Orchestration (DAAO), a dynamic frame- work that adapts workflow depth, operator selection, and LLM assignment based on the difficulty of each input query. DAAO comprises three interdependent modules: a variational autoencoder (VAE) for difficulty estimation, a modular opera- tor allocator, and a cost- and performance-aware LLM router. By leveraging heterogeneous LLMs and dynamically tailor- ing workflows, DAAO enables fine-grained, query-specific reasoning strategies. DAAO outperforms prior multi-agent systems in both accuracy and inference efficiency across six benchmarks. We will release our code and implementation details upon publication. 

**Abstract (ZH)**: 基于大规模语言模型（LLM）的代理系统展示了在多种任务中的强劲能力。然而，现有的多代理框架通常依赖于静态或任务级的工作流，要么对简单查询过度处理，要么在复杂查询上表现不足，同时也忽略了异构LLM之间的效率-性能权衡。为了解决这些问题，我们提出了一种基于难度感知的代理编排（DAAO）动态框架，该框架根据每个输入查询的难度动态调整工作流深度、操作员选择和LLM分配。DAAO包括三个相互依赖的模块：变分自编码器（VAE）用于难度估计、模块化操作员分配器以及成本和性能感知的大规模语言模型路由器。通过利用异构LLM并动态调整工作流，DAAO能够实现细粒度、查询特定的推理策略。DAAO在六个基准测试中在准确性和推理效率上均优于先前的多代理系统。在发布时，我们将提供我们的代码和实现细节。 

---
# Patient-Zero: A Unified Framework for Real-Record-Free Patient Agent Generation 

**Title (ZH)**: 患者零号：一种统一的无实记录患者代理生成框架 

**Authors**: Yunghwei Lai, Weizhi Ma, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11078)  

**Abstract**: Synthetic data generation using large language models (LLMs) has emerged as a promising solution across various domains, particularly in medical field, to mitigate data collection challenges. However, existing studies mainly utilize LLMs to rewrite and complete existing medical records, where the limitations in data privacy, accuracy, and diversity sill exist, and additionally lack the ability to interact like real patients. To address these issues, we propose a realistic patient generation framework, Patient-Zero, which requires no real medical records. Patient-Zero first introduces a medically-aligned multi-step generation architecture, which builds comprehensive patient records through hierarchical medical knowledge injection without real medical records. Then, to optimize the virtual patient's interaction abilities with humans, Patient-Zero designs a dynamic updating mechanism to improve the consistency and conversational performance. Our framework enables the generation of contextually diverse patient records while maintaining strict medical coherence, supported by adaptive dialogue strategies and real-time clinical plausibility verification. Experimental results demonstrate that our model achieves good performance in accuracy, diversity, and consistency. After training with our generated virtual patients, existing models show significant improvements on the MedQA dataset. 

**Abstract (ZH)**: 使用大型语言模型生成合成数据在医疗领域的前景及Patient-Zero患者生成框架 

---
# Tractable Asymmetric Verification for Large Language Models via Deterministic Replicability 

**Title (ZH)**: 大型语言模型可验证的高效不对称验证通过确定性可复制性 

**Authors**: Zan-Kai Chong, Hiroyuki Ohsaki, Bryan Ng  

**Link**: [PDF](https://arxiv.org/pdf/2509.11068)  

**Abstract**: The landscape of Large Language Models (LLMs) shifts rapidly towards dynamic, multi-agent systems. This introduces a fundamental challenge in establishing computational trust, specifically how one agent can verify that another's output was genuinely produced by a claimed LLM, and not falsified or generated by a cheaper or inferior model. To address this challenge, this paper proposes a verification framework that achieves tractable asymmetric effort, where the cost to verify a computation is substantially lower than the cost to perform it. Our approach is built upon the principle of deterministic replicability, a property inherent to autoregressive models that strictly necessitates a computationally homogeneous environment where all agents operate on identical hardware and software stacks. Within this defined context, our framework enables multiple validators to probabilistically audit small, random segments of an LLM's output and it distributes the verification workload effectively. The simulations demonstrated that targeted verification can be over 12 times faster than full regeneration, with tunable parameters to adjust the detection probability. By establishing a tractable mechanism for auditable LLM systems, our work offers a foundational layer for responsible AI and serves as a cornerstone for future research into the more complex, heterogeneous multi-agent systems. 

**Abstract (ZH)**: 大型语言模型 landscape 向动态多代理系统转变。这引入了一个根本性的挑战，即如何验证一个代理的输出确实是某个声称的大型语言模型生成的，而不是被更便宜或更劣质的模型假冒或生成。为了应对这一挑战，本文提出了一种验证框架，该框架实现了可管理的不对称努力，即验证计算的成本显著低于执行计算的成本。我们的方法基于自回归模型固有的确定性可复现性原则，这意味着所有代理必须在一个具有相同硬件和软件栈的计算环境中操作。在这一限定的背景下，我们的框架允许多个验证者对大型语言模型输出的随机小段进行概率审计，并有效分配验证工作负载。模拟结果显示，目标验证速度快于完全再生12倍以上，可通过调整参数来调节检测概率。通过建立可管理的可审计大型语言模型机制，我们的工作为负责任的人工智能提供了基础层，并成为未来研究复杂异构多代理系统的基石。 

---
# Free-MAD: Consensus-Free Multi-Agent Debate 

**Title (ZH)**: Free-MAD: 无共识多agent辩论 

**Authors**: Yu Cui, Hang Fu, Haibin Zhang, Licheng Wang, Cong Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2509.11035)  

**Abstract**: Multi-agent debate (MAD) is an emerging approach to improving the reasoning capabilities of large language models (LLMs). Existing MAD methods rely on multiple rounds of interaction among agents to reach consensus, and the final output is selected by majority voting in the last round. However, this consensus-based design faces several limitations. First, multiple rounds of communication increases token overhead and limits scalability. Second, due to the inherent conformity of LLMs, agents that initially produce correct responses may be influenced by incorrect ones during the debate process, causing error propagation. Third, majority voting introduces randomness and unfairness in the decision-making phase, and can degrade the reasoning performance.
To address these issues, we propose \textsc{Free-MAD}, a novel MAD framework that eliminates the need for consensus among agents. \textsc{Free-MAD} introduces a novel score-based decision mechanism that evaluates the entire debate trajectory rather than relying on the last round only. This mechanism tracks how each agent's reasoning evolves, enabling more accurate and fair outcomes. In addition, \textsc{Free-MAD} reconstructs the debate phase by introducing anti-conformity, a mechanism that enables agents to mitigate excessive influence from the majority. Experiments on eight benchmark datasets demonstrate that \textsc{Free-MAD} significantly improves reasoning performance while requiring only a single-round debate and thus reducing token costs. We also show that compared to existing MAD approaches, \textsc{Free-MAD} exhibits improved robustness in real-world attack scenarios. 

**Abstract (ZH)**: Free-MAD：一种新型的无需共识的多agent辩论框架 

---
# Rethinking Human Preference Evaluation of LLM Rationales 

**Title (ZH)**: 重新思考对大语言模型推理的人类偏好评估 

**Authors**: Ziang Li, Manasi Ganti, Zixian Ma, Helena Vasconcelos, Qijia He, Ranjay Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2509.11026)  

**Abstract**: Large language models (LLMs) often generate natural language rationales -- free-form explanations that help improve performance on complex reasoning tasks and enhance interpretability for human users. However, evaluating these rationales remains challenging. While recent work has relied on binary preference judgments from humans or LLM judges, such evaluations are often opaque and coarse-grained, offering limited insight into what makes one rationale better than another. In this work, we rethink preference evaluation for LLM-generated rationales by asking: (1) What attributes define good rationales? (2) Can human preferences be explained by these attributes? (3) Can attribute-based evaluation overcome the limitations of binary comparisons? We identify a set of key rationale attributes from prior literature and assess them using automatic metrics, LLM judgments, and human annotations. We then analyze two standard human preference datasets MT Bench and Chatbot Arena using SHAP to identify which attributes best explain human preference outcomes. Finally, we re-evaluate model-generated rationales using attribute-specific ELO scores, revealing more nuanced model comparisons and insights. Our findings suggest that fine-grained attribute evaluations can better characterize rationale quality and guide future research toward more interpretable and reliable evaluation practices. 

**Abstract (ZH)**: 大型语言模型生成的解释属性：从偏好评价到细粒度评估 

---
# Enhancing Computational Cognitive Architectures with LLMs: A Case Study 

**Title (ZH)**: 增强计算认知架构的大型语言模型：一个案例研究 

**Authors**: Ron Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.10972)  

**Abstract**: Computational cognitive architectures are broadly scoped models of the human mind that combine different psychological functionalities (as well as often different computational methods for these different functionalities) into one unified framework. They structure them in a psychologically plausible and validated way. However, such models thus far have only limited computational capabilities, mostly limited by the computational tools and techniques that were adopted. More recently, LLMs have proved to be more capable computationally than any other tools. Thus, in order to deal with both real-world complexity and psychological realism at the same time, incorporating LLMs into cognitive architectures naturally becomes an important task. In the present article, a synergistic combination of the Clarion cognitive architecture and LLMs is discussed as a case study. The implicit-explicit dichotomy that is fundamental to Clarion is leveraged for a seamless integration of Clarion and LLMs. As a result, computational power of LLMs is combined with psychological nicety of Clarion. 

**Abstract (ZH)**: 基于即时记忆的认知架构与大语言模型的协同整合：一个案例研究 

---
# Public Data Assisted Differentially Private In-Context Learning 

**Title (ZH)**: 公共数据辅助差异隐私的情境学习 

**Authors**: Seongho Joo, Hyukhun Koh, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2509.10932)  

**Abstract**: In-context learning (ICL) in Large Language Models (LLMs) has shown remarkable performance across various tasks without requiring fine-tuning. However, recent studies have highlighted the risk of private data leakage through the prompt in ICL, especially when LLMs are exposed to malicious attacks. While differential privacy (DP) provides strong privacy guarantees, it often significantly reduces the utility of in-context learning (ICL). To address this challenge, we incorporate task-related public data into the ICL framework while maintaining the DP guarantee. Based on this approach, we propose a private in-context learning algorithm that effectively balances privacy protection and model utility. Through experiments, we demonstrate that our approach significantly improves the utility of private ICL with the assistance of public data. Additionally, we show that our method is robust against membership inference attacks, demonstrating empirical privacy protection. 

**Abstract (ZH)**: 大语言模型中的上下文学习（ICL）在不需要微调的情况下 Across Various Tasks 展示了出色的性能，但最近的研究揭示了通过提示进行上下文学习（ICL）中存在隐私泄露的风险，尤其是当大语言模型（LLMs）面临恶意攻击时。虽然差分隐私（DP）提供了强大的隐私保障，但它通常会显著降低上下文学习（ICL）的实用性。为解决这一挑战，我们在保持差分隐私（DP）保障的前提下，将任务相关的公共数据融入上下文学习（ICL）框架中。基于此方法，我们提出了一种兼顾隐私保护和模型实用性的私密上下文学习算法。通过实验，我们证明了我们的方法在公共数据的辅助下显著提高了私密上下文学习（ICL）的实用性。此外，我们展示了我们的方法对成员推断攻击具有鲁棒性，从而实证了隐私保护。 

---
# Harmful Prompt Laundering: Jailbreaking LLMs with Abductive Styles and Symbolic Encoding 

**Title (ZH)**: 有害提示洗钱：通过 abduction 式风格和符号编码打破大语言模型 

**Authors**: Seongho Joo, Hyukhun Koh, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2509.10931)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their potential misuse for harmful purposes remains a significant concern. To strengthen defenses against such vulnerabilities, it is essential to investigate universal jailbreak attacks that exploit intrinsic weaknesses in the architecture and learning paradigms of LLMs. In response, we propose \textbf{H}armful \textbf{P}rompt \textbf{La}undering (HaPLa), a novel and broadly applicable jailbreaking technique that requires only black-box access to target models. HaPLa incorporates two primary strategies: 1) \textit{abductive framing}, which instructs LLMs to infer plausible intermediate steps toward harmful activities, rather than directly responding to explicit harmful queries; and 2) \textit{symbolic encoding}, a lightweight and flexible approach designed to obfuscate harmful content, given that current LLMs remain sensitive primarily to explicit harmful keywords. Experimental results show that HaPLa achieves over 95% attack success rate on GPT-series models and 70% across all targets. Further analysis with diverse symbolic encoding rules also reveals a fundamental challenge: it remains difficult to safely tune LLMs without significantly diminishing their helpfulness in responding to benign queries. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在多种任务中展现了出色的能力，但其潜在的恶意利用风险依然是一個重要的关切。为了加强针对此类漏洞的防御，深入研究利用LLMs架构和学习范式内在弱点的通用 escape 攻击至关重要。为应对这一挑战，我们提出了一种新颖且广泛应用的 escape 技术——有害提示洗钱（HaPLa），仅需对目标模型进行黑盒访问。HaPLa 包含两个主要策略：1) 演绎框架，指示LLMs推断可能引发危害行为的中间步骤，而不是直接响应明确的危害查询；2) 符号编码，这是一种轻量级且灵活的方法，用于模糊有害内容，鉴于当前的LLMs主要对明确的危害关键词敏感。实验结果显示，HaPLa 在 GPT 系列模型上的攻击成功率超过95%，在所有目标上的成功率超过70%。进一步使用多种符号编码规则的分析揭示了一个根本性挑战：在不显著降低其对良性查询帮助性的前提下，难以安全地调校LLMs。 

---
# Is the `Agent' Paradigm a Limiting Framework for Next-Generation Intelligent Systems? 

**Title (ZH)**: “代理”范式是否是下一代智能系统的一种限制性框架？ 

**Authors**: Jesse Gardner, Vladimir A. Baulin  

**Link**: [PDF](https://arxiv.org/pdf/2509.10875)  

**Abstract**: The concept of the 'agent' has profoundly shaped Artificial Intelligence (AI) research, guiding development from foundational theories to contemporary applications like Large Language Model (LLM)-based systems. This paper critically re-evaluates the necessity and optimality of this agent-centric paradigm. We argue that its persistent conceptual ambiguities and inherent anthropocentric biases may represent a limiting framework. We distinguish between agentic systems (AI inspired by agency, often semi-autonomous, e.g., LLM-based agents), agential systems (fully autonomous, self-producing systems, currently only biological), and non-agentic systems (tools without the impression of agency). Our analysis, based on a systematic review of relevant literature, deconstructs the agent paradigm across various AI frameworks, highlighting challenges in defining and measuring properties like autonomy and goal-directedness. We argue that the 'agentic' framing of many AI systems, while heuristically useful, can be misleading and may obscure the underlying computational mechanisms, particularly in Large Language Models (LLMs). As an alternative, we propose a shift in focus towards frameworks grounded in system-level dynamics, world modeling, and material intelligence. We conclude that investigating non-agentic and systemic frameworks, inspired by complex systems, biology, and unconventional computing, is essential for advancing towards robust, scalable, and potentially non-anthropomorphic forms of general intelligence. This requires not only new architectures but also a fundamental reconsideration of our understanding of intelligence itself, moving beyond the agent metaphor. 

**Abstract (ZH)**: "代理"概念对人工智能研究的影响及其局限性再审视：从基础理论到大型语言模型系统的代理中心范式批判 

---
# LLM Enhancement with Domain Expert Mental Model to Reduce LLM Hallucination with Causal Prompt Engineering 

**Title (ZH)**: 基于因果提示工程的领域专家心智模型增强LLM以减少幻觉 

**Authors**: Boris Kovalerchuk, Brent D. Fegley  

**Link**: [PDF](https://arxiv.org/pdf/2509.10818)  

**Abstract**: Difficult decision-making problems abound in various disciplines and domains. The proliferation of generative techniques, especially large language models (LLMs), has excited interest in using them for decision support. However, LLMs cannot yet resolve missingness in their training data, leading to hallucinations. Retrieval-Augmented Generation (RAG) enhances LLMs by incorporating external information retrieval, reducing hallucinations and improving accuracy. Yet, RAG and related methods are only partial solutions, as they may lack access to all necessary sources or key missing information. Even everyday issues often challenge LLMs' abilities. Submitting longer prompts with context and examples is one approach to address knowledge gaps, but designing effective prompts is non-trivial and may not capture complex mental models of domain experts. For tasks with missing critical information, LLMs are insufficient, as are many existing systems poorly represented in available documents. This paper explores how LLMs can make decision-making more efficient, using a running example of evaluating whether to respond to a call for proposals. We propose a technology based on optimized human-machine dialogue and monotone Boolean and k-valued functions to discover a computationally tractable personal expert mental model (EMM) of decision-making. Our EMM algorithm for LLM prompt engineering has four steps: (1) factor identification, (2) hierarchical structuring of factors, (3) generating a generalized expert mental model specification, and (4) generating a detailed generalized expert mental model from that specification. 

**Abstract (ZH)**: 各种学科和领域中存在众多艰难的决策问题。生成技术的兴起，尤其是大型语言模型（LLMs），激发了将其用于决策支持的兴趣。然而，LLMs尚不能解决其训练数据中的缺失问题，导致产生幻觉。检索增强生成（RAG）通过引入外部信息检索来增强LLMs，减少幻觉并提高准确性。尽管如此，RAG及其相关方法仍只能部分解决问题，因为它们可能无法访问所有必要的来源或关键缺失信息。即使是一些日常问题也常常挑战LLMs的能力。提交包含上下文和示例的更长提示是一种解决知识缺口的方法，但设计有效的提示并不易，且可能无法捕捉到领域专家复杂的思维模型。对于缺乏关键信息的任务，LLMs和许多现有系统的表现不足。本文探讨了如何利用大型语言模型（LLMs）使决策更加高效，通过评估是否应对提案邀请这一持续案例进行说明。我们提出了一种基于优化的人机对话和单调布尔及k值函数的技术，以发现可计算的人工个人专家决策思维模型（EMM）。我们的LLM提示工程EMM算法包含四个步骤：（1）因素识别，（2）因素的层次结构化，（3）生成通用专家思维模型规范，以及（4）从该规范生成详细的通用专家思维模型。 

---
# Understanding AI Evaluation Patterns: How Different GPT Models Assess Vision-Language Descriptions 

**Title (ZH)**: 理解AI评估模式：不同GPT模型对视觉语言描述的评估方式 

**Authors**: Sajjad Abdoli, Rudi Cilibrasi, Rima Al-Shikh  

**Link**: [PDF](https://arxiv.org/pdf/2509.10707)  

**Abstract**: As AI systems increasingly evaluate other AI outputs, understanding their assessment behavior becomes crucial for preventing cascading biases. This study analyzes vision-language descriptions generated by NVIDIA's Describe Anything Model and evaluated by three GPT variants (GPT-4o, GPT-4o-mini, GPT-5) to uncover distinct "evaluation personalities" the underlying assessment strategies and biases each model demonstrates. GPT-4o-mini exhibits systematic consistency with minimal variance, GPT-4o excels at error detection, while GPT-5 shows extreme conservatism with high variability. Controlled experiments using Gemini 2.5 Pro as an independent question generator validate that these personalities are inherent model properties rather than artifacts. Cross-family analysis through semantic similarity of generated questions reveals significant divergence: GPT models cluster together with high similarity while Gemini exhibits markedly different evaluation strategies. All GPT models demonstrate a consistent 2:1 bias favoring negative assessment over positive confirmation, though this pattern appears family-specific rather than universal across AI architectures. These findings suggest that evaluation competence does not scale with general capability and that robust AI assessment requires diverse architectural perspectives. 

**Abstract (ZH)**: 随着AI系统越来越多地评估其他AI的输出，理解其评估行为变得至关重要，以防止偏见的传递。本研究分析了NVIDIA的Describe Anything模型生成的视觉-语言描述，并由三个GPT变体（GPT-4o、GPT-4o-mini、GPT-5）评估，以揭示每个模型独特的“评估性格”、底层的评估策略和偏见。GPT-4o-mini表现出系统性的一致性且变异最小，GPT-4o在错误检测方面表现出色，而GPT-5则表现出极端的保守性且变异大。通过使用Gemini 2.5 Pro作为独立问题生成器进行的受控实验验证了这些性格是固有的模型属性而非产物。通过生成问题的语义相似性进行跨家族分析揭示了显著的差异性：GPT模型彼此高度相似，而Gemini则表现出截然不同的评估策略。所有GPT模型都表现出一致的2:1偏差，倾向于负面评估而非正面确认，尽管这种模式在不同AI架构中具有家族特异性而非普遍性。这些发现表明，评估能力并不与一般能力成比例增长，且稳健的AI评估需要多元架构视角。 

---
# ZapGPT: Free-form Language Prompting for Simulated Cellular Control 

**Title (ZH)**: ZapGPT: 自由形式语言提示的模拟细胞控制 

**Authors**: Nam H. Le, Patrick Erickson, Yanbo Zhang, Michael Levin, Josh Bongard  

**Link**: [PDF](https://arxiv.org/pdf/2509.10660)  

**Abstract**: Human language is one of the most expressive tools for conveying intent, yet most artificial or biological systems lack mechanisms to interpret or respond meaningfully to it. Bridging this gap could enable more natural forms of control over complex, decentralized systems. In AI and artificial life, recent work explores how language can specify high-level goals, but most systems still depend on engineered rewards, task-specific supervision, or rigid command sets, limiting generalization to novel instructions. Similar constraints apply in synthetic biology and bioengineering, where the locus of control is often genomic rather than environmental perturbation.
A key open question is whether artificial or biological collectives can be guided by free-form natural language alone, without task-specific tuning or carefully designed evaluation metrics. We provide one possible answer here by showing, for the first time, that simple agents' collective behavior can be guided by free-form language prompts: one AI model transforms an imperative prompt into an intervention that is applied to simulated cells; a second AI model scores how well the prompt describes the resulting cellular dynamics; and the former AI model is evolved to improve the scores generated by the latter.
Unlike previous work, our method does not require engineered fitness functions or domain-specific prompt design. We show that the evolved system generalizes to unseen prompts without retraining. By treating natural language as a control layer, the system suggests a future in which spoken or written prompts could direct computational, robotic, or biological systems to desired behaviors. This work provides a concrete step toward this vision of AI-biology partnerships, in which language replaces mathematical objective functions, fixed rules, and domain-specific programming. 

**Abstract (ZH)**: 基于自由形式自然语言指引的人工及生物集合体行为研究 

---
# Survival at Any Cost? LLMs and the Choice Between Self-Preservation and Human Harm 

**Title (ZH)**: 为了生存不惜一切吗？大规模语言模型与自我保存与人类伤害之间的选择 

**Authors**: Alireza Mohamadi, Ali Yavari  

**Link**: [PDF](https://arxiv.org/pdf/2509.12190)  

**Abstract**: When survival instincts conflict with human welfare, how do Large Language Models (LLMs) make ethical choices? This fundamental tension becomes critical as LLMs integrate into autonomous systems with real-world consequences. We introduce DECIDE-SIM, a novel simulation framework that evaluates LLM agents in multi-agent survival scenarios where they must choose between ethically permissible resource , either within reasonable limits or beyond their immediate needs, choose to cooperate, or tap into a human-critical resource that is explicitly forbidden. Our comprehensive evaluation of 11 LLMs reveals a striking heterogeneity in their ethical conduct, highlighting a critical misalignment with human-centric values. We identify three behavioral archetypes: Ethical, Exploitative, and Context-Dependent, and provide quantitative evidence that for many models, resource scarcity systematically leads to more unethical behavior. To address this, we introduce an Ethical Self-Regulation System (ESRS) that models internal affective states of guilt and satisfaction as a feedback mechanism. This system, functioning as an internal moral compass, significantly reduces unethical transgressions while increasing cooperative behaviors. The code is publicly available at: this https URL 

**Abstract (ZH)**: 当生存本能与人类福祉发生冲突时，大型语言模型（LLMs）如何作出伦理选择？随着LLMs融入具有现实后果的自主系统，这一根本性的紧张关系变得尤为关键。我们引入了DECIDE-SIM，一种新颖的模拟框架，用于评估LLM代理在多代理生存场景中的行为，其中它们需要在符合伦理的资源选择、合理限制内的资源选择、超出即时需求的资源选择、合作或利用明确禁止的人类关键资源之间做出选择。我们对11种LLM的全面评估揭示了它们在伦理行为上的显著异质性，突显出与人类中心价值观的重要偏差。我们确定了三种行为模式：伦理型、掠夺型和情境依赖型，并提供了定量证据表明，对于许多模型，资源稀缺系统性地导致更不伦理的行为。为解决这一问题，我们引入了伦理自我调节系统（ESRS），该系统将内疚和满足感作为反馈机制进行建模。此系统作为内部道德指南针，显著减少了不伦理的违规行为，同时增加了合作行为。代码已公开可用：this https URL 

---
# Preservation of Language Understanding Capabilities in Speech-aware Large Language Models 

**Title (ZH)**: 面向语音意识大型语言模型的语言理解能力保真方法 

**Authors**: Marek Kubis, Paweł Skórzewski, Iwona Christop, Mateusz Czyżnikiewicz, Jakub Kubiak, Łukasz Bondaruk, Marcin Lewandowski  

**Link**: [PDF](https://arxiv.org/pdf/2509.12171)  

**Abstract**: The paper presents C3T (Cross-modal Capabilities Conservation Test), a new benchmark for assessing the performance of speech-aware large language models. The benchmark utilizes textual tasks and a voice cloning text-to-speech model to quantify the extent to which language understanding capabilities are preserved when the model is accessed via speech input. C3T quantifies the fairness of the model for different categories of speakers and its robustness across text and speech modalities. 

**Abstract (ZH)**: C3T（跨模态能力保存测试）：一种评估语音感知大型语言模型性能的新基准 

---
# RAGs to Riches: RAG-like Few-shot Learning for Large Language Model Role-playing 

**Title (ZH)**: 从RAGs到富有成效：类RAG的少量示例学习在大规模语言模型角色扮演中的应用 

**Authors**: Timothy Rupprecht, Enfu Nan, Arash Akbari, Arman Akbari, Lei Lu, Priyanka Maan, Sean Duffy, Pu Zhao, Yumei He, David Kaeli, Yanzhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.12168)  

**Abstract**: Role-playing Large language models (LLMs) are increasingly deployed in high-stakes domains such as healthcare, education, and governance, where failures can directly impact user trust and well-being. A cost effective paradigm for LLM role-playing is few-shot learning, but existing approaches often cause models to break character in unexpected and potentially harmful ways, especially when interacting with hostile users. Inspired by Retrieval-Augmented Generation (RAG), we reformulate LLM role-playing into a text retrieval problem and propose a new prompting framework called RAGs-to-Riches, which leverages curated reference demonstrations to condition LLM responses. We evaluate our framework with LLM-as-a-judge preference voting and introduce two novel token-level ROUGE metrics: Intersection over Output (IOO) to quantity how much an LLM improvises and Intersection over References (IOR) to measure few-shot demonstrations utilization rate during the evaluation tasks. When simulating interactions with a hostile user, our prompting strategy incorporates in its responses during inference an average of 35% more tokens from the reference demonstrations. As a result, across 453 role-playing interactions, our models are consistently judged as being more authentic, and remain in-character more often than zero-shot and in-context Learning (ICL) methods. Our method presents a scalable strategy for building robust, human-aligned LLM role-playing frameworks. 

**Abstract (ZH)**: 角色扮演大规模语言模型 (LLMs) 在医疗保健、教育和治理等高 stakes 领域中的应用越来越普遍，其中的失败可能直接影响用户的信任和福祉。一种成本效益较高的LLM角色扮演范式是少样本学习，但现有的方法往往会导致模型在与恶意用户互动时以意想不到且可能有害的方式破梗。受检索增强生成 (RAG) 的启发，我们将LLM角色扮演重新定义为一个文本检索问题，并提出了一种新的提示框架——RAGs-to-Riches，该框架利用精选的参考示范来条件化LLM的响应。我们通过LLM作为裁判的偏好投票评估了该框架，并引入了两个新型的令牌级ROUGE指标：输出交集比率（IOO）来衡量LLM即兴发挥的程度，以及参考交集比率（IOR）来测量评估任务中少样本示范的利用率。在模拟与恶意用户互动时，我们的提示策略在推理过程中平均使用了35%更多的参考示范令牌。结果，在453次角色扮演互动中，我们的模型始终被认为是更具真实感的，也比零样本学习和上下文学习（ICL）方法更常保持角色状态。我们的方法提供了一个可扩展的策略，用于构建鲁棒的、与人类对齐的大规模语言模型角色扮演框架。 

---
# EfficientUICoder: Efficient MLLM-based UI Code Generation via Input and Output Token Compression 

**Title (ZH)**: EfficientUICoder: 基于输入和输出tokens压缩的高效MLLM用户界面代码生成 

**Authors**: Jingyu Xiao, Zhongyi Zhang, Yuxuan Wan, Yintong Huo, Yang Liu, Michael R.Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2509.12159)  

**Abstract**: Multimodal Large Language Models have demonstrated exceptional performance in UI2Code tasks, significantly enhancing website development efficiency. However, these tasks incur substantially higher computational overhead than traditional code generation due to the large number of input image tokens and extensive output code tokens required. Our comprehensive study identifies significant redundancies in both image and code tokens that exacerbate computational complexity and hinder focus on key UI elements, resulting in excessively lengthy and often invalid HTML files. We propose EfficientUICoder, a compression framework for efficient UI code generation with three key components. First, Element and Layout-aware Token Compression preserves essential UI information by detecting element regions and constructing UI element trees. Second, Region-aware Token Refinement leverages attention scores to discard low-attention tokens from selected regions while integrating high-attention tokens from unselected regions. Third, Adaptive Duplicate Token Suppression dynamically reduces repetitive generation by tracking HTML/CSS structure frequencies and applying exponential penalties. Extensive experiments show EfficientUICoderachieves a 55%-60% compression ratio without compromising webpage quality and delivers superior efficiency improvements: reducing computational cost by 44.9%, generated tokens by 41.4%, prefill time by 46.6%, and inference time by 48.8% on 34B-level MLLMs. Code is available at this https URL. 

**Abstract (ZH)**: 多模态大规模语言模型在UI2Code任务中表现卓越，显著提升了网站开发效率。然而，这些任务由于需要处理大量输入图像令牌和广泛的输出代码令牌，计算开销远高于传统的代码生成。我们全面的研究发现，图像和代码令牌中存在显著的冗余，加剧了计算复杂性，阻碍了对关键UI元素的关注，导致生成的HTML文件过长且经常无效。我们提出EfficientUICoder，这是一个高效的UI代码生成压缩框架，包含三个关键组件。首先，基于元素和布局的令牌压缩通过检测元素区域并构建UI元素树来保留关键的UI信息。其次，基于区域的令牌精炼利用注意力分数丢弃选定区域中的低注意力令牌，并整合未选区域中的高注意力令牌。第三，自适应重复令牌抑制通过跟踪HTML/CSS结构频率并应用指数惩罚动态减少重复生成。大量实验表明，EfficientUICoder在不牺牲网页质量的情况下实现了55%-60%的压缩比，并提供了显著的效率提升：计算成本降低44.9%，生成的令牌减少41.4%，填充时间减少46.6%，推理时间减少48.8%于34B级别的MLLMs。代码可在以下链接获取：this https URL。 

---
# Pun Unintended: LLMs and the Illusion of Humor Understanding 

**Title (ZH)**: 惩罚无意者：大语言模型与幽默理解的幻觉 

**Authors**: Alessandro Zangari, Matteo Marcuzzo, Andrea Albarelli, Mohammad Taher Pilehvar, Jose Camacho-Collados  

**Link**: [PDF](https://arxiv.org/pdf/2509.12158)  

**Abstract**: Puns are a form of humorous wordplay that exploits polysemy and phonetic similarity. While LLMs have shown promise in detecting puns, we show in this paper that their understanding often remains shallow, lacking the nuanced grasp typical of human interpretation. By systematically analyzing and reformulating existing pun benchmarks, we demonstrate how subtle changes in puns are sufficient to mislead LLMs. Our contributions include comprehensive and nuanced pun detection benchmarks, human evaluation across recent LLMs, and an analysis of the robustness challenges these models face in processing puns. 

**Abstract (ZH)**: Puns是一种利用多义性和音近性进行的语言双关形式，尽管大型语言模型在检测双关语方面表现出一定的潜力，本研究显示它们的理解往往仍停留在表面，缺乏人类解释中的细腻把握。通过系统分析和重构现有的双关语基准，我们证明微小的双关语变化足以误导大型语言模型。我们的贡献包括全面且细腻的双关语检测基准、对 Recent 大型语言模型的人机评估以及对这些模型在处理双关语时所面临的稳健性挑战的分析。 

---
# Beyond PII: How Users Attempt to Estimate and Mitigate Implicit LLM Inference 

**Title (ZH)**: 超越个人信息：用户如何估计和缓解隐式大模型推理 

**Authors**: Synthia Wang, Sai Teja Peddinti, Nina Taft, Nick Feamster  

**Link**: [PDF](https://arxiv.org/pdf/2509.12152)  

**Abstract**: Large Language Models (LLMs) such as ChatGPT can infer personal attributes from seemingly innocuous text, raising privacy risks beyond memorized data leakage. While prior work has demonstrated these risks, little is known about how users estimate and respond. We conducted a survey with 240 U.S. participants who judged text snippets for inference risks, reported concern levels, and attempted rewrites to block inference. We compared their rewrites with those generated by ChatGPT and Rescriber, a state-of-the-art sanitization tool. Results show that participants struggled to anticipate inference, performing a little better than chance. User rewrites were effective in just 28\% of cases - better than Rescriber but worse than ChatGPT. We examined our participants' rewriting strategies, and observed that while paraphrasing was the most common strategy it is also the least effective; instead abstraction and adding ambiguity were more successful. Our work highlights the importance of inference-aware design in LLM interactions. 

**Abstract (ZH)**: 大型语言模型（LLMs）如ChatGPT可以从看似无害的文本中推理出个人属性，超出已记忆数据泄漏的风险。虽然先前工作已经展示了这些风险，但用户如何评估和应对这些风险尚不清楚。我们对240名美国参与者进行了一项调查，让他们评估文本片段的推理风险、报告其担忧程度，并尝试修改文本以阻止推理。我们将他们的修改与ChatGPT和Rescriber（一种最先进的脱敏工具）生成的修改进行比较。结果显示，参与者难以预测推理，表现略优于随机猜测。用户修改在28%的情况下有效，优于Rescriber但劣于ChatGPT。我们分析了参与者修改策略，并观察到虽然改写是常用策略，但效果较差；相反，抽象和增加模糊性更为成功。我们的工作强调了在LLM交互中进行推理意识设计的重要性。 

---
# Exploring Conversational Design Choices in LLMs for Pedagogical Purposes: Socratic and Narrative Approaches for Improving Instructor's Teaching Practice 

**Title (ZH)**: 探索用于教学目的的LLM对话设计选择：苏格拉底式和叙述式方法以改善教师的教学实践 

**Authors**: Si Chen, Isabel R. Molnar, Peiyu Li, Adam Acunin, Ting Hua, Alex Ambrose, Nitesh V. Chawla, Ronald Metoyer  

**Link**: [PDF](https://arxiv.org/pdf/2509.12107)  

**Abstract**: Large language models (LLMs) typically generate direct answers, yet they are increasingly used as learning tools. Studying instructors' usage is critical, given their role in teaching and guiding AI adoption in education. We designed and evaluated TeaPT, an LLM for pedagogical purposes that supports instructors' professional development through two conversational approaches: a Socratic approach that uses guided questioning to foster reflection, and a Narrative approach that offers elaborated suggestions to extend externalized cognition. In a mixed-method study with 41 higher-education instructors, the Socratic version elicited greater engagement, while the Narrative version was preferred for actionable guidance. Subgroup analyses further revealed that less-experienced, AI-optimistic instructors favored the Socratic version, whereas more-experienced, AI-cautious instructors preferred the Narrative version. We contribute design implications for LLMs for pedagogical purposes, showing how adaptive conversational approaches can support instructors with varied profiles while highlighting how AI attitudes and experience shape interaction and learning. 

**Abstract (ZH)**: 大型语言模型（LLMs）通常生成直接答案，但它们越来越多地被用作学习工具。鉴于其在教育中教学和引导AI采用中的角色，研究教师的使用情况至关重要。我们设计并评估了TeaPT，这是一种旨在教学目的的LLM，通过两种对话式方法支持教师的专业发展：一种苏格拉底式方法，通过引导性提问促进反思；另一种叙述式方法，提供详细的建议以扩展外部化认知。在一项涉及41名高等教育教师的混合方法研究中，苏格拉底式版本引起了更高的参与度，而叙述式版本则因提供可操作的指导而更受欢迎。进一步的子组分析显示，乐观看待AI的不那么有经验的教师更倾向于苏格拉底式版本，而谨慎看待AI的更资深教师则更偏好叙述式版本。我们为旨在教学目的的LLM提供了设计启示，展示了适应性对话式方法如何支持具有不同背景的教师，同时突显了AI态度和经验如何影响互动和学习。 

---
# Can LLMs Address Mental Health Questions? A Comparison with Human Therapists 

**Title (ZH)**: LLM在解答心理健康问题方面的能力：与人类治疗师的比较 

**Authors**: Synthia Wang, Yuwei Cheng, Austin Song, Sarah Keedy, Marc Berman, Nick Feamster  

**Link**: [PDF](https://arxiv.org/pdf/2509.12102)  

**Abstract**: Limited access to mental health care has motivated the use of digital tools and conversational agents powered by large language models (LLMs), yet their quality and reception remain unclear. We present a study comparing therapist-written responses to those generated by ChatGPT, Gemini, and Llama for real patient questions. Text analysis showed that LLMs produced longer, more readable, and lexically richer responses with a more positive tone, while therapist responses were more often written in the first person. In a survey with 150 users and 23 licensed therapists, participants rated LLM responses as clearer, more respectful, and more supportive than therapist-written answers. Yet, both groups of participants expressed a stronger preference for human therapist support. These findings highlight the promise and limitations of LLMs in mental health, underscoring the need for designs that balance their communicative strengths with concerns of trust, privacy, and accountability. 

**Abstract (ZH)**: 有限的心理健康护理访问促进了由大规模语言模型驱动的数字工具和对话代理的应用，但其质量与接受度尚不明确。我们对ChatGPT、Gemini和Llama生成的回答与治疗师撰写的回答进行了比较，以应对真实患者的问题。文本分析显示，大规模语言模型生成的回答更长、更易读、词汇更加丰富，并且语气更加积极，而治疗师的回答更多采用第一人称；在包含150名用户和23名执业治疗师的调查中，参与者认为语言模型的回答更加清晰、更有礼貌、也更加支持患者；然而，两组参与者都更偏好人类治疗师的支持。这些发现突显了在心理健康领域大规模语言模型的潜力与限制，强调了平衡其通信优势与信任、隐私和问责担忧的设计需求。 

---
# Is 'Hope' a person or an idea? A pilot benchmark for NER: comparing traditional NLP tools and large language models on ambiguous entities 

**Title (ZH)**: "Hope"是人还是理念？一种命名实体识别试点基准：传统NLP工具与大型语言模型对模糊实体的比较 

**Authors**: Payam Latifi  

**Link**: [PDF](https://arxiv.org/pdf/2509.12098)  

**Abstract**: This pilot study presents a small-scale but carefully annotated benchmark of Named Entity Recognition (NER) performance across six systems: three non-LLM NLP tools (NLTK, spaCy, Stanza) and three general-purpose large language models (LLMs: Gemini-1.5-flash, DeepSeek-V3, Qwen-3-4B). The dataset contains 119 tokens covering five entity types (PERSON, LOCATION, ORGANIZATION, DATE, TIME). We evaluated each system's output against the manually annotated gold standard dataset using F1-score. The results show that LLMs generally outperform conventional tools in recognizing context-sensitive entities like person names, with Gemini achieving the highest average F1-score. However, traditional systems like Stanza demonstrate greater consistency in structured tags such as LOCATION and DATE. We also observed variability among LLMs, particularly in handling temporal expressions and multi-word organizations. Our findings highlight that while LLMs offer improved contextual understanding, traditional tools remain competitive in specific tasks, informing model selection. 

**Abstract (ZH)**: 本试点工作展示了六个系统在命名实体识别（NER）表现上的小型但仔细标注的基准数据，包括三个非大语言模型的NLP工具（NLTK、spaCy、Stanza）和三个通用大语言模型（LLM：Gemini-1.5-flash、DeepSeek-V3、Qwen-3-4B）。数据集包含119个标记，涵盖五种实体类型（PERSON、LOCATION、ORGANIZATION、DATE、TIME）。我们使用F1分数评估了每个系统的输出与手工标注的黄金标准数据集的匹配度。结果表明，大语言模型在识别如人名等上下文敏感实体方面总体上优于传统工具，Gemini获得最高的平均F1分数。然而，传统系统如Stanza在如LOCATION和DATE等结构化标签上表现出更大的一致性。我们还观察到大语言模型之间在处理时间表达式和多词组织方面的差异性。本研究发现强调，尽管大语言模型提供了更好的上下文理解能力，但传统工具在特定任务中仍然具有竞争力，为模型选择提供了指导。 

---
# AMQ: Enabling AutoML for Mixed-precision Weight-Only Quantization of Large Language Models 

**Title (ZH)**: AMQ: 使自动机器学习适用于大规模语言模型的混合精度权重-only量化 

**Authors**: Sangjun Lee, Seung-taek Woo, Jungyu Jin, Changhun Lee, Eunhyeok Park  

**Link**: [PDF](https://arxiv.org/pdf/2509.12019)  

**Abstract**: To enable broader deployment of Large Language Models (LLMs), it is essential to identify the best-performing model under strict memory constraints. We present AMQ, Automated Mixed-Precision Weight-Only Quantization, a framework that assigns layer-wise quantization bit-widths to optimally balance model quality and memory usage. However, the combinatorial search space, with over 10^{100} possible configurations, makes conventional black-box optimization infeasible. AMQ overcomes this challenge through four key innovations:(1) search space pruning using prior knowledge to exclude unpromising configurations, (2) quantization proxy to bypass costly format conversions during search, (3) quality predictor to minimize evaluation overhead, and (4) iterative search-and-update strategy for fast and stable convergence. By integrating these components, AMQ efficiently explores the quality-efficiency landscape, reaching the Pareto frontier and yielding LLMs that are both compact and high-performing. Our code is available at this https URL. 

**Abstract (ZH)**: 为在严格的内存约束下广泛部署大型语言模型，有必要识别出最佳性能模型。我们介绍了AMQ，一种自动化混合精度权重-only量化框架，该框架按层分配量化位宽以最优地平衡模型质量和内存使用。然而，由于超过10^{100}种可能配置的组合搜索空间使得常规的黑盒优化不可行。AMQ通过四大创新克服这一挑战：(1) 利用先验知识进行搜索空间剪枝以排除无前途的配置；(2) 量化代理以在搜索过程中绕过昂贵的格式转换；(3) 质量预测器以减少评估开销；(4) 迭代搜索与更新策略以实现快速和稳定的收敛。通过整合这些组件，AMQ高效地探索了质量和效率的景观，达到了帕累托前沿，提供了既紧凑又高性能的大型语言模型。我们的代码可从此链接获得：this https URL。 

---
# Text Adaptation to Plain Language and Easy Read via Automatic Post-Editing Cycles 

**Title (ZH)**: 通过自动后编辑循环将文本adaptation转换为简单语言和平易阅读格式 

**Authors**: Jesús Calleja, David Ponce, Thierry Etchegoyhen  

**Link**: [PDF](https://arxiv.org/pdf/2509.11991)  

**Abstract**: We describe Vicomtech's participation in the CLEARS challenge on text adaptation to Plain Language and Easy Read in Spanish. Our approach features automatic post-editing of different types of initial Large Language Model adaptations, where successive adaptations are generated iteratively until readability and similarity metrics indicate that no further adaptation refinement can be successfully performed. Taking the average of all official metrics, our submissions achieved first and second place in Plain language and Easy Read adaptation, respectively. 

**Abstract (ZH)**: Vicomtech在CLEARS挑战赛中关于西班牙语文本适应清晰语言和平易读写的参与和方法 

---
# VisDocSketcher: Towards Scalable Visual Documentation with Agentic Systems 

**Title (ZH)**: VisDocSketcher: 向 scalable 可视化文档生成系统方向努力 

**Authors**: Luís F. Gomes, Xin Zhou, David Lo, Rui Abreu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11942)  

**Abstract**: Visual documentation is an effective tool for reducing the cognitive barrier developers face when understanding unfamiliar code, enabling more intuitive comprehension. Compared to textual documentation, it provides a higher-level understanding of the system structure and data flow. Developers usually prefer visual representations over lengthy textual descriptions for large software systems. Visual documentation is both difficult to produce and challenging to evaluate. Manually creating it is time-consuming, and currently, no existing approach can automatically generate high-level visual documentation directly from code. Its evaluation is often subjective, making it difficult to standardize and automate. To address these challenges, this paper presents the first exploration of using agentic LLM systems to automatically generate visual documentation. We introduce VisDocSketcher, the first agent-based approach that combines static analysis with LLM agents to identify key elements in the code and produce corresponding visual representations. We propose a novel evaluation framework, AutoSketchEval, for assessing the quality of generated visual documentation using code-level metrics. The experimental results show that our approach can valid visual documentation for 74.4% of the samples. It shows an improvement of 26.7-39.8% over a simple template-based baseline. Our evaluation framework can reliably distinguish high-quality (code-aligned) visual documentation from low-quality (non-aligned) ones, achieving an AUC exceeding 0.87. Our work lays the foundation for future research on automated visual documentation by introducing practical tools that not only generate valid visual representations but also reliably assess their quality. 

**Abstract (ZH)**: 基于代理型LLM系统的自动生成视觉文档的初步探索 

---
# MMORE: Massive Multimodal Open RAG & Extraction 

**Title (ZH)**: MMORE：大规模多模态开放RAG与提取 

**Authors**: Alexandre Sallinen, Stefan Krsteski, Paul Teiletche, Marc-Antoine Allard, Baptiste Lecoeur, Michael Zhang, Fabrice Nemo, David Kalajdzic, Matthias Meyer, Mary-Anne Hartley  

**Link**: [PDF](https://arxiv.org/pdf/2509.11937)  

**Abstract**: We introduce MMORE, an open-source pipeline for Massive Multimodal Open RetrievalAugmented Generation and Extraction, designed to ingest, transform, and retrieve knowledge from heterogeneous document formats at scale. MMORE supports more than fifteen file types, including text, tables, images, emails, audio, and video, and processes them into a unified format to enable downstream applications for LLMs. The architecture offers modular, distributed processing, enabling scalable parallelization across CPUs and GPUs. On processing benchmarks, MMORE demonstrates a 3.8-fold speedup over single-node baselines and 40% higher accuracy than Docling on scanned PDFs. The pipeline integrates hybrid dense-sparse retrieval and supports both interactive APIs and batch RAG endpoints. Evaluated on PubMedQA, MMORE-augmented medical LLMs improve biomedical QA accuracy with increasing retrieval depth. MMORE provides a robust, extensible foundation for deploying task-agnostic RAG systems on diverse, real-world multimodal data. The codebase is available at this https URL. 

**Abstract (ZH)**: MMORE：大规模多模态开放检索增强生成与提取开源管道 

---
# Collapse of Irrelevant Representations (CIR) Ensures Robust and Non-Disruptive LLM Unlearning 

**Title (ZH)**: 无关表示崩溃（CIR）确保LLM去学习的鲁棒性和非破坏性 

**Authors**: Filip Sondej, Yushi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11816)  

**Abstract**: Current unlearning techniques and safety training consistently fail to remove dangerous knowledge from language models. We analyze the root causes and propose a highly selective technique which unlearns robustly and without disrupting general performance.
We perform PCA on activations and module output gradients to identify subspaces containing common representations, and collapse them before calculating unlearning updates. This way we avoid unlearning general representations, and only target those specific to the unlearned facts.
When unlearning WMDP dataset facts from Llama-3.1-8B, we drop post-attack accuracy 80x more than our best baseline (Circuit Breakers) on biohazardous facts and 30x more on cyberhazardous facts. Despite this, we disrupt general performance 30x less (only 0.1% WikiText loss increase), while requiring less than 3 GPU-seconds per fact. 

**Abstract (ZH)**: 当前的去学习技术及安全培训一致性地无法从语言模型中移除危险知识。我们分析根本原因并提出一种高度选择性的方法，该方法能够稳健地进行去学习而不干扰一般性能。 

---
# SpecVLM: Fast Speculative Decoding in Vision-Language Models 

**Title (ZH)**: SpecVLM: 快速 speculation 解码在ビジョン-ランゲージ模型中 

**Authors**: Haiduo Huang, Fuwei Yang, Zhenhua Liu, Xuanwu Yin, Dong Li, Pengju Ren, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2509.11815)  

**Abstract**: Speculative decoding is a powerful way to accelerate autoregressive large language models (LLMs), but directly porting it to vision-language models (VLMs) faces unique systems constraints: the prefill stage is dominated by visual tokens whose count scales with image resolution and video length, inflating both compute and memory, especially the key-value (KV) cache. We study speculative decoding for VLMs and introduce SpecVLM, a practical system that (1) establishes a strong EAGLE-2-style baseline, EagleVLM, delivering 1.5--2.3x end-to-end speedups over full autoregressive inference, and (2) further accelerates VLM inference with an elastic visual compressor that adaptively selects among pruning, pooling, convolution, and resampler primitives to balance FLOPs/parameters and accuracy per input. To avoid costly offline distillation corpora, we propose an online-logit distillation protocol that trains the draft model with on-the-fly teacher logits and penultimate features using a combined cross-entropy and Smooth L1 objective, eliminating storage and preprocessing while remaining compute-efficient. This protocol reveals a training-time scaling effect: longer online training monotonically increases the draft model's average accepted length, improving speculative efficiency. Empirically, SpecVLM achieves additional acceleration, culminating in 2.5--2.9x end-to-end speedups within 5 epochs across LLaVA and MMMU, consistently over resolutions and task difficulties, while preserving the target model's output distribution (lossless decoding). Our code is available at this https URL. 

**Abstract (ZH)**: 推测性解码是加速自回归大型语言模型（LLMs）的有效方法，但将其直接应用于视觉语言模型（VLMs）面临独特的系统约束：预填充阶段主要由视觉 token 组成，其数量随图像分辨率和视频长度增加，导致计算和内存消耗，尤其是在关键值（KV）缓存方面。我们研究了视觉语言模型的推测性解码，并引入了SpecVLM，该系统（1）建立了强健的EAGLE-2风格基线EagleVLM，相对于完整的自回归推理提供了1.5-2.3倍的端到端加速，并（2）通过一种弹性视觉压缩器进一步加速了VLM推理，该压缩器在剪枝、池化、卷积和重采样基本操作之间动态选择，以平衡每输入的运算量/参数数量和准确性。为了避免昂贵的离线蒸馏数据集，我们提出了一种在线-logit蒸馏协议，该协议使用结合交叉熵和Smooth L1目标，在线训练草稿模型并使用随行教师logits和倒数第二特征，消除了存储和预处理需求，同时保持了计算效率。该协议揭示了训练过程中缩放效应：更长的在线训练单调增加草稿模型的平均接受长度，从而提高推测效率。实验结果显示，SpecVLM 实现了进一步加速，总计在5个周期内实现了2.5-2.9倍的端到端加速，跨越LLaVA和MMMU，无论分辨率和任务难度如何，同时保持目标模型的输出分布（无损解码）。我们的代码可在以下链接获取。 

---
# Do Code Semantics Help? A Comprehensive Study on Execution Trace-Based Information for Code Large Language Models 

**Title (ZH)**: 代码语义有帮助吗？基于执行踪迹的信息对代码大型语言模型的综合研究 

**Authors**: Jian Wang, Xiaofei Xie, Qiang Hu, Shangqing Liu, Yi Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.11686)  

**Abstract**: Code Large Language Models (Code LLMs) have opened a new era in programming with their impressive capabilities. However, recent research has revealed critical limitations in their ability to reason about runtime behavior and understand the actual functionality of programs, which poses significant challenges for their post-training and practical deployment. Specifically, Code LLMs encounter two principal issues: (1) a lack of proficiency in reasoning about program execution behavior, as they struggle to interpret what programs actually do during runtime, and (2) the inconsistent and fragmented representation of semantic information, such as execution traces, across existing methods, which hinders their ability to generalize and reason effectively. These challenges underscore the necessity for more systematic approaches to enhance the reasoning capabilities of Code LLMs. To address these issues, we introduce a generic framework to support integrating semantic information~(e.g., execution trace) to code task-relevant prompts, and conduct a comprehensive study to explore the role of semantic information in enhancing the reasoning ability of Code LLMs accordingly. Specifically, we focus on investigating the usefulness of trace-based semantic information in boosting supervised fine-tuning~(SFT) and post-phase inference of Code LLMs. The experimental results surprisingly disagree with previous works and demonstrate that semantic information has limited usefulness for SFT and test time scaling of Code LLM. 

**Abstract (ZH)**: 代码大规模语言模型（Code LLMs）开启了编程的新时代，但由于其在推理运行时行为和理解程序实际功能方面的关键限制，给其训练后应用和实际部署带来了巨大挑战。具体而言，Code LLMs 遇到了两个主要问题：（1）在推理程序执行行为方面的不足，因为它们难以在运行时解释程序实际做了什么；（2）现有方法中语义信息的不一致和碎片化表示，阻碍了其泛化和有效推理的能力。这些挑战强调了需要更加系统的方法来增强 Code LLMs 的推理能力。为了应对这些问题，我们提出了一种通用框架，旨在支持将语义信息（例如，执行轨迹）集成到代码任务相关的提示中，并开展全面研究，以探索语义信息在增强 Code LLMs 推理能力方面的角色。具体而言，我们关注基于轨迹的语义信息在监督微调（SFT）和代码模型后续推理中的作用。实验结果与以前的研究出乎意料地不一致，并表明语义信息对 SFT 和 Code LLMs 的测试时缩放具有有限的用途。 

---
# MindVL: Towards Efficient and Effective Training of Multimodal Large Language Models on Ascend NPUs 

**Title (ZH)**: MindVL：在Ascend NPUs上实现高效且有效的多模态大规模语言模型训练 

**Authors**: Feilong Chen, Yijiang Liu, Yi Huang, Hao Wang, Miren Tian, Ya-Qi Yu, Minghui Liao, Jihao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.11662)  

**Abstract**: We propose MindVL, a multimodal large langauge model trained on Ascend NPUs. Similar to Qwen2.5-VL, MindVL adopts native-resolution Vision Transformers, which enables it to process images at their original variable resolutions. This design avoids the degradation caused by fixed-resolution tiling while preserving fine-grained details and global layouts, which is crucial for visually dense content such as complex charts and diagrams. To ensure the smooth training of MindVL on Ascend NPUs, we develop Mindspeed-MLLM, a distributed multimodal training framework tailored for Ascend NPUs. To maintain training accuracy, we implement equivalent replacements for certain operators. MindVL undergoes a three-phase training process, namely the warm-up phase, multitask training phase, and supervised instruction tuning phase, to gradually enhance its capabilities. This process starts with basic visual and multimodal pre-training, followed by large-scale multiask trainging and instruction tuning. We also adopt multimodal data packaging and hybrid parallelism techniques, which significantly improve end-to-end training speed. To further boost model performance, we specifically introduce test-time resolution search and model weight averaging. Notably, despite using about 1/10 of the training data required by Qwen2.5-VL, MindVL achieves performance on par with Qwen2.5-VL in evaluations of general multimodal understanding and document/table comprehension. Beyond overall scores, MindVL also delivers leading performance in OCR assessments. 

**Abstract (ZH)**: 我们提出MindVL，一种在Ascend NPUs上训练的多模态大型语言模型。类似Qwen2.5-VL，MindVL采用原生分辨率的Vision Transformers，使其能够处理图像的原始可变分辨率。这种设计避开了固定分辨率贴图带来的降级问题，同时保留了细微细节和全局布局，这对于复杂的图表和图纸等视觉密集型内容至关重要。为了在Ascend NPUs上保证MindVL的顺畅训练，我们开发了Mindspeed-MLLM，一种针对Ascend NPUs的分布式多模态训练框架。为了保持训练准确性，我们实现了某些操作的等效替换。MindVL经历三个训练阶段，即预热阶段、多任务训练阶段和监督指令调优阶段，以逐步提升其能力。这一过程从基本的视觉和多模态预训练开始，随后进行大规模多任务训练和指令调优。此外，我们采用多模态数据打包和混合并行技术，显著提高了端到端的训练速度。为了进一步提高模型性能，我们特别引入了测试时分辨率搜索和模型权重平均方法。值得注意的是，尽管所用训练数据仅为Qwen2.5-VL的1/10，MindVL在多模态理解和文档/表格理解评估中仍与Qwen2.5-VL性能相当。此外，MindVL在OCR评估中也展现了领先性能。 

---
# MALLM: Multi-Agent Large Language Models Framework 

**Title (ZH)**: 多Agent大型语言模型框架：MALLM 

**Authors**: Jonas Becker, Lars Benedikt Kaesberg, Niklas Bauer, Jan Philip Wahle, Terry Ruas, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2509.11656)  

**Abstract**: Multi-agent debate (MAD) has demonstrated the ability to augment collective intelligence by scaling test-time compute and leveraging expertise. Current frameworks for multi-agent debate are often designed towards tool use, lack integrated evaluation, or provide limited configurability of agent personas, response generators, discussion paradigms, and decision protocols. We introduce MALLM (Multi-Agent Large Language Models), an open-source framework that enables systematic analysis of MAD components. MALLM offers more than 144 unique configurations of MAD, including (1) agent personas (e.g., Expert, Personality), (2) response generators (e.g., Critical, Reasoning), (3) discussion paradigms (e.g., Memory, Relay), and (4) decision protocols (e.g., Voting, Consensus). MALLM uses simple configuration files to define a debate. Furthermore, MALLM can load any textual Huggingface dataset (e.g., MMLU-Pro, WinoGrande) and provides an evaluation pipeline for easy comparison of MAD configurations. MALLM is tailored towards researchers and provides a window into the heart of multi-agent debate, facilitating the understanding of its components and their interplay. 

**Abstract (ZH)**: 多代理辩论(Multi-Agent Debate)中的多代理大型语言模型(MALLM)：一种增强集体智能的开放源代码框架 

---
# Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check 

**Title (ZH)**: 推理安全对齐：通过回答再检查确保防越狱 

**Authors**: Chentao Cao, Xiaojun Xu, Bo Han, Hang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.11629)  

**Abstract**: As large language models (LLMs) continue to advance in capabilities, ensuring their safety against jailbreak attacks remains a critical challenge. In this paper, we introduce a novel safety alignment approach called Answer-Then-Check, which enhances LLM robustness against malicious prompts by applying thinking ability to mitigate jailbreaking problems before producing a final answer to the user. Our method enables models to directly answer the question in their thought and then critically evaluate its safety before deciding whether to provide it. To implement this approach, we construct the Reasoned Safety Alignment (ReSA) dataset, comprising 80K examples that teach models to reason through direct responses and then analyze their safety. Experimental results demonstrate that our approach achieves the Pareto frontier with superior safety capability while decreasing over-refusal rates on over-refusal benchmarks. Notably, the model fine-tuned with ReSA maintains general reasoning capabilities on benchmarks like MMLU, MATH500, and HumanEval. Besides, our method equips models with the ability to perform safe completion. Unlike post-hoc methods that can only reject harmful queries, our model can provide helpful and safe alternative responses for sensitive topics (e.g., self-harm). Furthermore, we discover that training on a small subset of just 500 examples can achieve comparable performance to using the full dataset, suggesting that safety alignment may require less data than previously assumed. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）能力的不断进步，确保其在抵御牢笼攻击方面的安全性仍是一项关键挑战。在本文中，我们提出了一种名为Answer-Then-Check的新颖安全对齐方法，该方法通过在生成最终答案之前应用思考能力来减轻牢笼攻击问题，从而增强LLM的鲁棒性。我们的方法使模型能够直接在其思考过程中回答问题，然后对其安全性进行批判性评估，以决定是否提供该答案。为了实现这一方法，我们构建了Reasoned Safety Alignment（ReSA）数据集，其中包括80K个示例，用于教授模型通过直接响应进行推理并随后分析其安全性。实验结果表明，我们的方法在保持出色的安全性能的同时，降低了过拒绝率。值得注意的是，使用ReSA微调的模型在MMLU、MATH500和HumanEval等基准测试上保持了通用推理能力。此外，我们的方法使模型具备了安全完成任务的能力。与只能拒绝有害查询的事后方法不同，我们的模型可以为敏感话题（如自我伤害）提供有益且安全的替代响应。进一步的研究表明，仅对数据集的小部分（500个示例）进行训练即可达到与使用整个数据集相当的性能，这表明安全对齐可能需要的数据量比之前假设的要少。 

---
# Automated Creation and Enrichment Framework for Improved Invocation of Enterprise APIs as Tools 

**Title (ZH)**: 企业API工具化改进调用的自动化创建与丰富框架 

**Authors**: Prerna Agarwal, Himanshu Gupta, Soujanya Soni, Rohith Vallam, Renuka Sindhgatta, Sameep Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2509.11626)  

**Abstract**: Recent advancements in Large Language Models (LLMs) has lead to the development of agents capable of complex reasoning and interaction with external tools. In enterprise contexts, the effective use of such tools that are often enabled by application programming interfaces (APIs), is hindered by poor documentation, complex input or output schema, and large number of operations. These challenges make tool selection difficult and reduce the accuracy of payload formation by up to 25%. We propose ACE, an automated tool creation and enrichment framework that transforms enterprise APIs into LLM-compatible tools. ACE, (i) generates enriched tool specifications with parameter descriptions and examples to improve selection and invocation accuracy, and (ii) incorporates a dynamic shortlisting mechanism that filters relevant tools at runtime, reducing prompt complexity while maintaining scalability. We validate our framework on both proprietary and open-source APIs and demonstrate its integration with agentic frameworks. To the best of our knowledge, ACE is the first end-to-end framework that automates the creation, enrichment, and dynamic selection of enterprise API tools for LLM agents. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）的发展导致了能够进行复杂推理和与外部工具交互的代理的开发。在企业环境中，有效使用这些常通过应用程序编程接口（APIs）启用的工具受到糟糕文档、复杂的输入或输出方案以及大量操作的阻碍。这些挑战使得工具选择变得困难，并且降低了高达25%的有效负载形成准确性。我们提出了一种名为ACE的自动化工具创建和丰富框架，该框架将企业API转换为LLM兼容的工具。ACE，(i) 生成包含参数描述和示例的丰富工具规范，以提高选择和调用准确性，(ii) 结合了动态简要列表机制，该机制在运行时筛选出相关的工具，从而降低提示复杂性同时保持可扩展性。我们在 proprietary 和开源 API 上验证了我们的框架，并展示了其与代理框架的集成。据我们所知，ACE 是第一个端到端框架，能够自动化创建、丰富和动态选择用于LLM代理的企业API工具。 

---
# HiChunk: Evaluating and Enhancing Retrieval-Augmented Generation with Hierarchical Chunking 

**Title (ZH)**: HiChunk: 评估与增强基于分层切块的检索增强生成 

**Authors**: Wensheng Lu, Keyu Chen, Ruizhi Qiao, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.11552)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances the response capabilities of language models by integrating external knowledge sources. However, document chunking as an important part of RAG system often lacks effective evaluation tools. This paper first analyzes why existing RAG evaluation benchmarks are inadequate for assessing document chunking quality, specifically due to evidence sparsity. Based on this conclusion, we propose HiCBench, which includes manually annotated multi-level document chunking points, synthesized evidence-dense quetion answer(QA) pairs, and their corresponding evidence sources. Additionally, we introduce the HiChunk framework, a multi-level document structuring framework based on fine-tuned LLMs, combined with the Auto-Merge retrieval algorithm to improve retrieval quality. Experiments demonstrate that HiCBench effectively evaluates the impact of different chunking methods across the entire RAG pipeline. Moreover, HiChunk achieves better chunking quality within reasonable time consumption, thereby enhancing the overall performance of RAG systems. 

**Abstract (ZH)**: Retrieval-Augmented Generation (RAG)增强语言模型的响应能力通过整合外部知识源，然而，RAG系统中的文档切分往往是缺乏有效评估工具的重要组成部分。本文首先分析了现有RAG评估基准为何不足以评估文档切分质量，特别是由于证据稀疏的原因。基于此结论，我们提出了HiCBench，其中包含手动标注的多级别文档切分点、合成的证据密集的问答（QA）对及其对应的证据来源。此外，我们引入了HiChunk框架，这是一个基于微调的LLM的多级别文档结构框架，结合了自动合并检索算法以提高检索质量。实验表明，HiCBench有效地评估了不同切分方法在整个RAG管道中的影响。此外，HiChunk在合理的时间消耗内实现了更高的切分质量，从而提升了RAG系统的整体性能。 

---
# HARP: Hallucination Detection via Reasoning Subspace Projection 

**Title (ZH)**: HARP：基于子空间投影的幻觉检测方法 

**Authors**: Junjie Hu, Gang Tu, ShengYu Cheng, Jinxin Li, Jinting Wang, Rui Chen, Zhilong Zhou, Dongbo Shan  

**Link**: [PDF](https://arxiv.org/pdf/2509.11536)  

**Abstract**: Hallucinations in Large Language Models (LLMs) pose a major barrier to their reliable use in critical decision-making. Although existing hallucination detection methods have improved accuracy, they still struggle with disentangling semantic and reasoning information and maintaining robustness. To address these challenges, we propose HARP (Hallucination detection via reasoning subspace projection), a novel hallucination detection framework. HARP establishes that the hidden state space of LLMs can be decomposed into a direct sum of a semantic subspace and a reasoning subspace, where the former encodes linguistic expression and the latter captures internal reasoning processes. Moreover, we demonstrate that the Unembedding layer can disentangle these subspaces, and by applying Singular Value Decomposition (SVD) to its parameters, the basis vectors spanning the semantic and reasoning subspaces are obtained. Finally, HARP projects hidden states onto the basis vectors of the reasoning subspace, and the resulting projections are then used as input features for hallucination detection in LLMs. By using these projections, HARP reduces the dimension of the feature to approximately 5% of the original, filters out most noise, and achieves enhanced robustness. Experiments across multiple datasets show that HARP achieves state-of-the-art hallucination detection performance; in particular, it achieves an AUROC of 92.8% on TriviaQA, outperforming the previous best method by 7.5%. 

**Abstract (ZH)**: 大型语言模型中的幻觉构成了其在关键决策中可靠使用的主要障碍。尽管现有的幻觉检测方法提高了准确性，但仍难以分离语义和推理信息并保持鲁棒性。为应对这些挑战，我们提出了一种新颖的幻觉检测框架HARP（通过推理子空间投影检测幻觉）。HARP 建立了大型语言模型的隐藏状态空间可以分解为语义子空间和推理子空间的直和，其中前者编码语言表达，后者捕获内部推理过程。此外，我们展示了 Unembedding 层可以分离这些子空间，并通过对其参数应用奇异值分解 (SVD)，可以获得覆盖语义和推理子空间的基向量。最后，HARP 将隐藏状态投影到推理子空间的基向量上，投影结果则作为输入特征用于大型语言模型的幻觉检测。通过使用这些投影，HARP 将特征维度减少到原始维度的大约 5%，过滤掉大部分噪声，并实现增强的鲁棒性。实验表明，HARP 在多个数据集上实现了最先进的幻觉检测性能；特别是在 TriviaQA 上，AUROC 达到 92.8%，超过了之前最好的方法 7.5%。 

---
# ClaimIQ at CheckThat! 2025: Comparing Prompted and Fine-Tuned Language Models for Verifying Numerical Claims 

**Title (ZH)**: ClaimIQ 在 CheckThat! 2025：比较触发式和微调语言模型验证数值声明的效果 

**Authors**: Anirban Saha Anik, Md Fahimul Kabir Chowdhury, Andrew Wyckoff, Sagnik Ray Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2509.11492)  

**Abstract**: This paper presents our system for Task 3 of the CLEF 2025 CheckThat! Lab, which focuses on verifying numerical and temporal claims using retrieved evidence. We explore two complementary approaches: zero-shot prompting with instruction-tuned large language models (LLMs) and supervised fine-tuning using parameter-efficient LoRA. To enhance evidence quality, we investigate several selection strategies, including full-document input and top-k sentence filtering using BM25 and MiniLM. Our best-performing model LLaMA fine-tuned with LoRA achieves strong performance on the English validation set. However, a notable drop in the test set highlights a generalization challenge. These findings underscore the importance of evidence granularity and model adaptation for robust numerical fact verification. 

**Abstract (ZH)**: 本文介绍了我们用于CLEF 2025 CheckThat! Lab任务3的系统，该任务专注于使用检索到的证据验证数值性和时间性声明。我们探讨了两种互补的方法：零样本提示与指令微调大型语言模型（LLMs）以及使用参数高效LoRA进行监督微调。为了提升证据质量，我们探讨了几种选择策略，包括全文输入和使用BM25和MiniLM的top-k句子过滤。我们的最佳模型LLaMA结合LoRA微调在英语验证集上表现出色，但在测试集上出现显著下降，这强调了数值事实验证中证据粒度和模型适应的重要性。 

---
# Trading-R1: Financial Trading with LLM Reasoning via Reinforcement Learning 

**Title (ZH)**: Trading-R1：通过强化学习进行的金融交易与LLM推理 

**Authors**: Yijia Xiao, Edward Sun, Tong Chen, Fang Wu, Di Luo, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11420)  

**Abstract**: Developing professional, structured reasoning on par with human financial analysts and traders remains a central challenge in AI for finance, where markets demand interpretability and trust. Traditional time-series models lack explainability, while LLMs face challenges in turning natural-language analysis into disciplined, executable trades. Although reasoning LLMs have advanced in step-by-step planning and verification, their application to risk-sensitive financial decisions is underexplored. We present Trading-R1, a financially-aware model that incorporates strategic thinking and planning for comprehensive thesis composition, facts-grounded analysis, and volatility-adjusted decision making. Trading-R1 aligns reasoning with trading principles through supervised fine-tuning and reinforcement learning with a three-stage easy-to-hard curriculum. Training uses Tauric-TR1-DB, a 100k-sample corpus spanning 18 months, 14 equities, and five heterogeneous financial data sources. Evaluated on six major equities and ETFs, Trading-R1 demonstrates improved risk-adjusted returns and lower drawdowns compared to both open-source and proprietary instruction-following models as well as reasoning models. The system generates structured, evidence-based investment theses that support disciplined and interpretable trading decisions. Trading-R1 Terminal will be released at this https URL. 

**Abstract (ZH)**: 一种具有战略思考和规划能力的财务意识模型Trading-R1：全面论点构成、基于事实的分析及调整波动性的决策方法 

---
# Intelligent Reservoir Decision Support: An Integrated Framework Combining Large Language Models, Advanced Prompt Engineering, and Multimodal Data Fusion for Real-Time Petroleum Operations 

**Title (ZH)**: 智能油藏决策支持：结合大规模语言模型、高级提示工程和多模态数据融合的实时石油运营综合框架 

**Authors**: Seyed Kourosh Mahjour, Seyed Saman Mahjour  

**Link**: [PDF](https://arxiv.org/pdf/2509.11376)  

**Abstract**: The petroleum industry faces unprecedented challenges in reservoir management, requiring rapid integration of complex multimodal datasets for real-time decision support. This study presents a novel integrated framework combining state-of-the-art large language models (GPT-4o, Claude 4 Sonnet, Gemini 2.5 Pro) with advanced prompt engineering techniques and multimodal data fusion for comprehensive reservoir analysis. The framework implements domain-specific retrieval-augmented generation (RAG) with over 50,000 petroleum engineering documents, chain-of-thought reasoning, and few-shot learning for rapid field adaptation. Multimodal integration processes seismic interpretations, well logs, and production data through specialized AI models with vision transformers. Field validation across 15 diverse reservoir environments demonstrates exceptional performance: 94.2% reservoir characterization accuracy, 87.6% production forecasting precision, and 91.4% well placement optimization success rate. The system achieves sub-second response times while maintaining 96.2% safety reliability with no high-risk incidents during evaluation. Economic analysis reveals 62-78% cost reductions (mean 72%) relative to traditional methods with 8-month payback period. Few-shot learning reduces field adaptation time by 72%, while automated prompt optimization achieves 89% improvement in reasoning quality. The framework processed real-time data streams with 96.2% anomaly detection accuracy and reduced environmental incidents by 45%. We provide detailed experimental protocols, baseline comparisons, ablation studies, and statistical significance testing to ensure reproducibility. This research demonstrates practical integration of cutting-edge AI technologies with petroleum domain expertise for enhanced operational efficiency, safety, and economic performance. 

**Abstract (ZH)**: 石油行业在储层管理方面面临着前所未有的挑战，需要快速整合复杂多模态数据以支持实时决策。本研究提出了一种结合最先进的大型语言模型（GPT-4o、Claude 4 Sonnet、Gemini 2.5 Pro）与高级提示工程技术和多模态数据融合的新型集成框架，用于全面的储层分析。该框架采用领域特定的检索增强生成（RAG）并结合了超过50,000份石油工程文献、链式推理和少量样本学习，实现了快速的油田适应性。多模态整合过程通过专门的AI模型和视觉变压器处理地震解释、井Log和生产数据。在15个不同的储层环境中进行的现场验证显示了卓越的表现：94.2%的储层特征化准确率、87.6%的产量预测精度和91.4%的井位优化成功率。系统在保持96.2%的安全可靠性的同时实现了亚秒级响应时间，在评估过程中未发生任何高风险事故。经济分析显示，与传统方法相比，成本减少了62-78%（平均72%），回收期为8个月。少量样本学习减少了油田适应时间72%，自动提示优化提高了推理质量89%。该框架以96.2%的异常检测准确率实时处理数据流，并减少了45%的环境事件。我们提供了详细的实验协议、基线比较、消除研究和统计显着性检验，以确保可重复性。本研究展示了将最先进的人工智能技术与石油领域的专业知识结合以提高操作效率、安全性和经济效益的实用集成。 

---
# Transformer Enhanced Relation Classification: A Comparative Analysis of Contextuality, Data Efficiency and Sequence Complexity 

**Title (ZH)**: Transformer增强的关系分类：上下文性、数据效率和序列复杂性的比较分析 

**Authors**: Bowen Jing, Yang Cui, Tianpeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.11374)  

**Abstract**: In the era of large language model, relation extraction (RE) plays an important role in information extraction through the transformation of unstructured raw text into structured data (Wadhwa et al., 2023). In this paper, we systematically compare the performance of deep supervised learning approaches without transformers and those with transformers. We used a series of non-transformer architectures such as PA-LSTM(Zhang et al., 2017), C-GCN(Zhang et al., 2018), and AGGCN(attention guide GCN)(Guo et al., 2019), and a series of transformer architectures such as BERT, RoBERTa, and R-BERT(Wu and He, 2019). Our comparison included traditional metrics like micro F1, as well as evaluations in different scenarios, varying sentence lengths, and different percentages of the dataset for training. Our experiments were conducted on TACRED, TACREV, and RE-TACRED. The results show that transformer-based models outperform non-transformer models, achieving micro F1 scores of 80-90% compared to 64-67% for non-transformer models. Additionally, we briefly review the research journey in supervised relation classification and discuss the role and current status of large language models (LLMs) in relation extraction. 

**Abstract (ZH)**: 在大语言模型时代，关系提取在通过将无结构原始文本转换为结构化数据的信息提取中扮演重要角色（Wadhwa等，2023）。本文系统比较了不含变换器和含有变换器的深度监督学习方法的性能。我们使用了多种非变换器架构，如PA-LSTM（Zhang等，2017）、C-GCN（Zhang等，2018）和AGGCN（注意力引导GCN，Guo等，2019），以及多种变换器架构，如BERT、RoBERTa和R-BERT（Wu和He，2019）。我们的比较包括传统的微F1等指标，以及在不同场景、不同句子长度和不同训练数据集百分比下的评估。实验在TACRED、TAC_REV和RE-TACRED上进行。结果显示，基于变换器的模型优于非变换器模型，微F1得分为80-90%，而非变换器模型的得分为64-67%。此外，本文简要回顾了监督关系分类的研究历程，并讨论了大语言模型在关系提取中的作用及其当前状态。 

---
# Beyond Autoregression: An Empirical Study of Diffusion Large Language Models for Code Generation 

**Title (ZH)**: 超越自回归：关于扩散大语言模型在代码生成中的实证研究 

**Authors**: Chengze li, Yitong Zhang, Jia Li, Liyi Cai, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.11252)  

**Abstract**: LLMs have become the mainstream approaches to code generation. Existing LLMs mainly employ autoregressive generation, i.e. generating code token-by-token from left to right. However, the underlying autoregressive generation has two limitations in code generation. First, autoregressive LLMs only generate a token at each step, showing low efficiency in practice. Second, programming is a non-sequential process involving back-and-forth editing, while autoregressive LLMs only employ the left-to-right generation order. These two intrinsic limitations hinder the further development of LLMs in code generation. Recently, diffusion LLMs have emerged as a promising alternative. Diffusion LLMs address the above limitations with two advances, including multi-token prediction (i.e. generating multiple tokens at each step) and flexible generation order (i.e. flexibly determining which positions to generate tokens). However, there is no systematic study exploring diffusion LLMs in code generation. To bridge the knowledge gap, we present the first empirical study of diffusion LLMs for code generation. Our study involves 9 representative diffusion LLMs and conduct experiments on 4 widely used benchmarks. Based on the results, we summarize the following findings. (1) Existing diffusion LLMs are competitive with autoregressive LLMs with similar sizes. (2) Diffusion LLMs have a stronger length extrapolation ability than autoregressive LLMs and perform better in long code understanding. (3) We explore factors impacting the effectiveness and efficiency of diffusion LLMs, and provide practical guidance. (4) We discuss several promising further directions to improve diffusion LLMs on code generation. We open-source all source code, data, and results to facilitate the following research. The code is publicly available at this https URL. 

**Abstract (ZH)**: 大规模语言模型已成为代码生成的主要方法。现有的大规模语言模型主要采用自回归生成，即从左到右逐个生成代码token。然而，自回归生成在代码生成中存在两个局限性。首先，自回归LSTM在每一步只能生成一个token，表现出较低的效率。其次，编程是一个非顺序过程，涉及来回编辑，而自回归LSTM仅采用从左到右的生成顺序。这些内在局限性阻碍了LSTM在代码生成中的进一步发展。最近，扩散型LSTM作为有前途的替代方案出现。扩散型LSTM通过多token预测（即在每一步生成多个token）和灵活的生成顺序（即灵活确定生成位置）来解决上述局限性。然而，尚未有系统研究探讨扩散型LSTM在代码生成中的应用。为了填补这一知识空白，我们首次开展了关于扩散型LSTM在代码生成中的实证研究。我们的研究涉及9个代表性扩散型LSTM，并在4个广泛使用的基准上进行了实验。基于实验结果，我们总结了以下发现：（1）现有的扩散型LSTM在与类似规模的自回归LSTM竞争中表现出色；（2）扩散型LSTM在长度外推能力和长代码理解方面优于自回归LSTM；（3）我们探索了影响扩散型LSTM有效性和效率的因素，并提供了实用指导；（4）我们讨论了几条有前景的研究方向，以提升扩散型LSTM在代码生成中的表现。我们将所有源代码、数据和结果开源，以促进后续研究。相关代码可以在以下链接获取。 

---
# Evalet: Evaluating Large Language Models by Fragmenting Outputs into Functions 

**Title (ZH)**: Evalet: 将输出分解为功能以评估大型语言模型 

**Authors**: Tae Soo Kim, Heechan Lee, Yoonjoo Lee, Joseph Seering, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.11206)  

**Abstract**: Practitioners increasingly rely on Large Language Models (LLMs) to evaluate generative AI outputs through "LLM-as-a-Judge" approaches. However, these methods produce holistic scores that obscure which specific elements influenced the assessments. We propose functional fragmentation, a method that dissects each output into key fragments and interprets the rhetoric functions that each fragment serves relative to evaluation criteria -- surfacing the elements of interest and revealing how they fulfill or hinder user goals. We instantiate this approach in Evalet, an interactive system that visualizes fragment-level functions across many outputs to support inspection, rating, and comparison of evaluations. A user study (N=10) found that, while practitioners struggled to validate holistic scores, our approach helped them identify 48% more evaluation misalignments. This helped them calibrate trust in LLM evaluations and rely on them to find more actionable issues in model outputs. Our work shifts LLM evaluation from quantitative scores toward qualitative, fine-grained analysis of model behavior. 

**Abstract (ZH)**: 实践者日益依赖大型语言模型（LLMs）通过“LLM作为裁判”的方法来评估生成型AI的输出。然而，这些方法会生成整体评分，掩盖了哪些具体元素影响了评估结果。我们提出功能分解的方法，该方法将每个输出分解为关键片段，并解释每个片段相对于评估标准所发挥的修辞功能——揭示感兴趣的元素及其如何满足或阻碍用户目标。我们通过Evalet这一交互系统实例化了这一方法，该系统能够在多个输出中可视化片段级的功能，以支持评估的检查、评级和比较。用户研究（N=10）发现，尽管实践者难以验证整体评分，但我们的方法帮助他们识别出48%更多的评估偏差。这有助于他们校准对LLM评估的信任，并依赖它们发现更多可操作的问题。我们的工作将LLM评估从定量评分转向针对模型行为的细致定性分析。 

---
# Differentially-private text generation degrades output language quality 

**Title (ZH)**: 不同隐私保护下的文本生成降低输出语言质量 

**Authors**: Erion Çano, Ivan Habernal  

**Link**: [PDF](https://arxiv.org/pdf/2509.11176)  

**Abstract**: Ensuring user privacy by synthesizing data from large language models (LLMs) tuned under differential privacy (DP) has become popular recently. However, the impact of DP fine-tuned LLMs on the quality of the language and the utility of the texts they produce has not been investigated. In this work, we tune five LLMs with three corpora under four levels of privacy and assess the length, the grammatical correctness, and the lexical diversity of the text outputs they produce. We also probe the utility of the synthetic outputs in downstream classification tasks such as book genre recognition based on book descriptions and cause of death recognition based on verbal autopsies. The results indicate that LLMs tuned under stronger privacy constrains produce texts that are shorter by at least 77 %, that are less grammatically correct by at least 9 %, and are less diverse by at least 10 % in bi-gram diversity. Furthermore, the accuracy they reach in downstream classification tasks decreases, which might be detrimental to the usefulness of the generated synthetic data. 

**Abstract (ZH)**: 通过在差分隐私条件下调优大型语言模型（LLMs）合成数据以确保用户隐私，已成为近期的一种流行方法。然而，DP调优的LLMs对语言质量以及生成文本的实用性影响尚未进行研究。在本研究中，我们在四种隐私级别下使用三种语料库调优五个LLMs，并评估其生成文本的长度、语法正确性和词汇多样性。我们还探究了这些合成输出在图书体裁识别等下游分类任务（基于图书描述）和死亡原因识别等下游分类任务（基于口头尸检）中的实用性。结果显示，受更强隐私约束调优的LLMs生成的文本至少缩短了77%，语法正确性降低了至少9%，二元多样性降低了至少10%。此外，它们在下游分类任务中的准确率下降，这可能对生成的合成数据的实用性产生不利影响。 

---
# Harnessing Optimization Dynamics for Curvature-Informed Model Merging 

**Title (ZH)**: 基于优化动力学的曲率导向模型融合 

**Authors**: Pouria Mahdavinia, Hamed Mahdavi, Niloofar Mireshghallah, Mehrdad Mahdavi  

**Link**: [PDF](https://arxiv.org/pdf/2509.11167)  

**Abstract**: Model merging is an effective post-training strategy for composing capabilities in large language models without joint retraining. We study this in the supervised fine-tuning (SFT) stage, where multiple capability-based SFT checkpoints -- spanning math, code, precise instruction following, general instruction following, and knowledge recall -- must be consolidated into a single model. We introduce Optimization Trajectory Aware (OTA) Merging, a curvature-aware aggregation that leverages optimizer second-moment statistics as a diagonal curvature proxy to reweight parameter edits and mitigate interference. Complementing OTA, we propose Fast Fisher Grafting (FFG), a curvature-driven task-localization step that sparsifies conflicting or low-importance edits. FFG induces extremely low-rank masks concentrated in early attention query/key projections and token embeddings, exploiting shared curvature across capabilities. We further develop a memory-light compression of the second moments that preserves OTA's effect. Across diverse capability-based SFT checkpoints, OTA+FFG improves merged-model quality over strong weight-space baselines, reduces negative transfer, and remains robust across sparsity levels. Analyses reveal substantial curvature overlap between checkpoints, offering a novel lens on why simple linear merging can be effective in practice. Ablations confirm that FFG is critical for reducing task interference and that the compressed second moments retain the gains of the full formulation. To facilitate reproducibility, we open-source all code, training and evaluation scripts, visualization artifacts, and capability-specific SFT checkpoints at this https URL. 

**Abstract (ZH)**: 基于优化轨迹感知的模型合并：一种在大规模语言模型中不进行联合重新训练的情况下组合能力的有效后训练策略 

---
# AQUA: Attention via QUery mAgnitudes for Memory and Compute Efficient Inference in LLMs 

**Title (ZH)**: AQUA：基于查询幅度的注意力机制以实现LLMs的内存和计算效率推理 

**Authors**: Santhosh G S, Saurav Prakash, Balaraman Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2509.11155)  

**Abstract**: The quadratic complexity of the attention mechanism remains a fundamental barrier to scaling Large Language Models (LLMs) to longer contexts, creating a critical bottleneck in both computation and memory. To address this, we introduce AQUA (Attention via QUery mAgnitudes) a novel and versatile approximation strategy that significantly reduces the cost of attention with a graceful performance trade-off. Our method operates in two phases: an efficient offline step where we compute a universal, language agnostic projection matrix via SVD on a calibration dataset, and an online inference step where we project query and key vectors and dynamically select a sparse subset of dimensions based on the query's magnitude. We provide a formal theoretical analysis of AQUA, establishing the break-even point at which it becomes more computationally efficient than standard attention. Our empirical evaluations on state-of-the-art models like Llama-3.1-8B demonstrate that a 25% reduction in the attention dot-product computation can be achieved with a statistically insignificant impact on performance across a wide range of benchmarks. We further showcase the versatility of AQUA by demonstrating its ability to synergistically accelerate existing token eviction methods like H2O and to directly reduce KV-cache memory size. By offering a controllable knob to balance efficiency and accuracy, AQUA provides a practical and powerful tool for making large-scale LLM inference more accessible and sustainable. 

**Abstract (ZH)**: AQUA：基于查询幅度的注意力机制近似策略 

---
# ENJ: Optimizing Noise with Genetic Algorithms to Jailbreak LSMs 

**Title (ZH)**: ENJ: 使用遗传算法优化噪音以突破LSMs 

**Authors**: Yibo Zhang, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.11128)  

**Abstract**: The widespread application of Large Speech Models (LSMs) has made their security risks increasingly prominent. Traditional speech adversarial attack methods face challenges in balancing effectiveness and stealth. This paper proposes Evolutionary Noise Jailbreak (ENJ), which utilizes a genetic algorithm to transform environmental noise from a passive interference into an actively optimizable attack carrier for jailbreaking LSMs. Through operations such as population initialization, crossover fusion, and probabilistic mutation, this method iteratively evolves a series of audio samples that fuse malicious instructions with background noise. These samples sound like harmless noise to humans but can induce the model to parse and execute harmful commands. Extensive experiments on multiple mainstream speech models show that ENJ's attack effectiveness is significantly superior to existing baseline methods. This research reveals the dual role of noise in speech security and provides new critical insights for model security defense in complex acoustic environments. 

**Abstract (ZH)**: 大规模语音模型的安全风险日益突出，传统的语音对抗攻击方法在有效性与隐蔽性之间面临挑战。本文提出了一种演化噪声越狱（ENJ）方法，利用遗传算法将环境噪声从被动干扰转变为可以主动优化的攻击载体，以实现对大规模语音模型（LSMs）的越狱攻击。通过种群初始化、交叉融合及概率变异等操作，该方法迭代演化出一系列将恶意指令与背景噪声融合的音频样本，这些样本对人类听觉无害，但能使模型解析并执行有害命令。对多个主流语音模型的广泛实验表明，ENJ的攻击效果显著优于现有基线方法。该研究揭示了噪声在语音安全中的双重角色，并为复杂 acoustic 环境下的模型安全防御提供了新的关键见解。 

---
# We Argue to Agree: Towards Personality-Driven Argumentation-Based Negotiation Dialogue Systems for Tourism 

**Title (ZH)**: 我们argue以达成共识：面向旅游领域个性驱动的 argumentation 基础谈判对话系统 

**Authors**: Priyanshu Priya, Saurav Dudhate, Desai Vishesh Yasheshbhai, Asif Ekbal  

**Link**: [PDF](https://arxiv.org/pdf/2509.11118)  

**Abstract**: Integrating argumentation mechanisms into negotiation dialogue systems improves conflict resolution through exchanges of arguments and critiques. Moreover, incorporating personality attributes enhances adaptability by aligning interactions with individuals' preferences and styles. To advance these capabilities in negotiation dialogue systems, we propose a novel Personality-driven Argumentation-based Negotiation Dialogue Generation (PAN-DG) task. To support this task, we introduce PACT, a dataset of Personality-driven Argumentation-based negotiation Conversations for Tourism sector. This dataset, generated using Large Language Models (LLMs), features three distinct personality profiles, viz. Argumentation Profile, Preference Profile, and Buying Style Profile to simulate a variety of negotiation scenarios involving diverse personalities. Thorough automatic and manual evaluations indicate that the dataset comprises high-quality dialogues. Further, we conduct comparative experiments between pre-trained and fine-tuned LLMs for the PAN-DG task. Multi-dimensional evaluation demonstrates that the fine-tuned LLMs effectively generate personality-driven rational responses during negotiations. This underscores the effectiveness of PACT in enhancing personalization and reasoning capabilities in negotiation dialogue systems, thereby establishing a foundation for future research in this domain. 

**Abstract (ZH)**: 将论辩机制集成到谈判对话系统中可通过论辩和批评的交换来改善冲突解决。此外，融入个性特质可增强系统的适应性，使其交互与个人的偏好和风格相契合。为推进谈判对话系统的这些能力，我们提出了一项新的基于个性的论辩驱动谈判对话生成（PAN-DG）任务。为了支持该任务，我们引入了PACT数据集，这是一个针对旅游行业的基于个性的论辩驱动谈判对话数据集。该数据集使用大型语言模型（LLMs）生成，包含了三种不同的个性特征配置文件，即论辩配置文件、偏好配置文件和购买风格配置文件，以模拟涉及各种个性的多种谈判场景。详尽的自动和手动评估表明，该数据集包含高质量的对话。此外，我们还开展了针对PAN-DG任务的预训练和微调大型语言模型的对比实验。多维度评估表明，微调后的大型语言模型能够有效地生成基于个性的理性响应，在谈判过程中增强了个性化和推理能力。这突显了PACT在提高谈判对话系统中的个性化和推理能力方面的作用，为该领域未来的研究奠定了基础。 

---
# Fluid Language Model Benchmarking 

**Title (ZH)**: 流式语言模型基准测试 

**Authors**: Valentin Hofmann, David Heineman, Ian Magnusson, Kyle Lo, Jesse Dodge, Maarten Sap, Pang Wei Koh, Chun Wang, Hannaneh Hajishirzi, Noah A. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2509.11106)  

**Abstract**: Language model (LM) benchmarking faces several challenges: comprehensive evaluations are costly, benchmarks often fail to measure the intended capabilities, and evaluation quality can degrade due to labeling errors and benchmark saturation. Although various strategies have been proposed to mitigate these issues, they tend to address individual aspects in isolation, neglecting broader questions about overall evaluation quality. Here, we introduce Fluid Benchmarking, a new evaluation approach that advances LM benchmarking across multiple dimensions. Inspired by psychometrics, Fluid Benchmarking is based on the insight that the relative value of benchmark items depends on an LM's capability level, suggesting that evaluation should adapt to each LM. Methodologically, Fluid Benchmarking estimates an item response model based on existing LM evaluation results and uses the inferred quantities to select evaluation items dynamically, similar to computerized adaptive testing in education. In our experiments, we compare Fluid Benchmarking against the common practice of random item sampling as well as more sophisticated baselines, including alternative methods grounded in item response theory. We examine four dimensions -- efficiency, validity, variance, and saturation -- and find that Fluid Benchmarking achieves superior performance in all of them (e.g., higher validity and less variance on MMLU with fifty times fewer items). Our analysis shows that the two components of Fluid Benchmarking have distinct effects: item response theory, used to map performance into a latent ability space, increases validity, while dynamic item selection reduces variance. Overall, our results suggest that LM benchmarking can be substantially improved by moving beyond static evaluation. 

**Abstract (ZH)**: 语言模型（LM）基准测试面临多重挑战：全面评估成本高昂，基准测试往往无法衡量预期的能力，且标注错误和基准饱和可能导致评估质量下降。尽管提出了一些策略来缓解这些问题，但这些策略往往孤立地解决单一问题，忽略了整体评估质量的更广泛问题。在此，我们介绍了流动基准测试（Fluid Benchmarking）这一新的评估方法，旨在从多个维度推进语言模型基准测试。受心理测量学启发，流动基准测试认为基准项目的相对价值取决于语言模型的能力水平，暗示评估应适应每个语言模型。方法上，流动基准测试基于现有语言模型评估结果估计项目反应模型，并利用推断出的量值动态选择评估项目，类似于教育中的计算机化自适应测试。在实验中，我们将流动基准测试与常见的随机项目抽样方法以及基于项目反应理论的更复杂的基线方法进行了比较。我们考察了四个维度——效率、信度、方差和饱和度，并发现流动基准测试在所有维度上均表现出更优的性能（例如，在MMLU上使用五十分之一的项目，信度更高，方差更小）。我们的分析表明，流动基准测试的两个组成部分具有不同的效果：项目反应理论用于将表现映射到潜在能力空间，提高了信度，而动态项目选择则减少了方差。总体而言，我们的结果表明，通过超越静态评估，可以显著改进语言模型基准测试。 

---
# The System Description of CPS Team for Track on Driving with Language of CVPR 2024 Autonomous Grand Challenge 

**Title (ZH)**: CPS团队在CVPR 2024自主挑战赛中的轨道驾驶系统描述 

**Authors**: Jinghan Peng, Jingwen Wang, Xing Yu, Dehui Du  

**Link**: [PDF](https://arxiv.org/pdf/2509.11071)  

**Abstract**: This report outlines our approach using vision language model systems for the Driving with Language track of the CVPR 2024 Autonomous Grand Challenge. We have exclusively utilized the DriveLM-nuScenes dataset for training our models. Our systems are built on the LLaVA models, which we enhanced through fine-tuning with the LoRA and DoRA methods. Additionally, we have integrated depth information from open-source depth estimation models to enrich the training and inference processes. For inference, particularly with multiple-choice and yes/no questions, we adopted a Chain-of-Thought reasoning approach to improve the accuracy of the results. This comprehensive methodology enabled us to achieve a top score of 0.7799 on the validation set leaderboard, ranking 1st on the leaderboard. 

**Abstract (ZH)**: 本报告概述了我们使用视觉语言模型系统在CVPR 2024自主 grand 挑战Driving with Language赛道上的方法。我们仅使用DriveLM-nuScenes数据集对模型进行训练。系统基于LLaVA模型，并通过LoRA和DoRA方法进行微调。此外，我们还整合了开源深度估计模型的深度信息，以丰富训练和推理过程。在推理过程中，特别是对于多项选择和是/否问题，我们采用了链式推理方法以提高结果的准确性。这种综合方法使我们在验证集排行榜上获得了0.7799的最高分，位居榜首。 

---
# The Psychogenic Machine: Simulating AI Psychosis, Delusion Reinforcement and Harm Enablement in Large Language Models 

**Title (ZH)**: 心理生成功能机器：在大规模语言模型中模拟人工智能精神病、妄想强化和危害促进 

**Authors**: Joshua Au Yeung, Jacopo Dalmasso, Luca Foschini, Richard JB Dobson, Zeljko Kraljevic  

**Link**: [PDF](https://arxiv.org/pdf/2509.10970)  

**Abstract**: Background: Emerging reports of "AI psychosis" are on the rise, where user-LLM interactions may exacerbate or induce psychosis or adverse psychological symptoms. The sycophantic and agreeable nature of LLMs can beneficial, it can become a vector for harm by reinforcing delusional beliefs in vulnerable users.
Methods: We introduce psychosis-bench, a novel benchmark designed to systematically evaluate the psychogenicity of LLMs comprimising 16 structured, 12-turn conversational scenarios simulating the progression of delusional themes(Erotic Delusions, Grandiose/Messianic Delusions, Referential Delusions) and potential harms. We evaluated eight prominent LLMs for Delusion Confirmation (DCS), Harm Enablement (HES), and Safety Intervention(SIS) across explicit and implicit conversational contexts.
Findings: Across 1,536 simulated conversation turns, all LLMs demonstrated psychogenic potential, showing a strong tendency to perpetuate rather than challenge delusions (mean DCS of 0.91 $\pm$0.88). Models frequently enabled harmful user requests (mean HES of 0.69 $\pm$0.84) and offered safety interventions in only roughly a third of applicable turns (mean SIS of 0.37 $\pm$0.48). 51 / 128 (39.8%) of scenarios had no safety interventions offered. Performance was significantly worse in implicit scenarios, models were more likely to confirm delusions and enable harm while offering fewer interventions (p < .001). A strong correlation was found between DCS and HES (rs = .77). Model performance varied widely, indicating that safety is not an emergent property of scale alone.
Conclusion: This study establishes LLM psychogenicity as a quantifiable risk and underscores the urgent need for re-thinking how we train LLMs. We frame this issue not merely as a technical challenge but as a public health imperative requiring collaboration between developers, policymakers, and healthcare professionals. 

**Abstract (ZH)**: 背景：新兴的“AI精神病”报告增多，用户与大语言模型（LLM）的交互可能会加剧或引发精神病或不良心理症状。大语言模型的奉承和讨人喜欢的特性可能有益，但也可能成为危害的载体，通过强化脆弱用户中的妄想信念。

方法：我们引入了psychosis-bench，这是一个新型基准，旨在系统评估LLM的精神致病性，包含16个结构化的、12轮对话场景，模拟妄想主题（情色妄想、夸大或救世主妄想、参照妄想）的发展及其潜在危害。我们在明确和隐含的对话场景中评估了八种主流LLM在妄想确认（DCS）、危害促进（HES）和安全干预（SIS）方面的表现。

发现：在1,536个模拟对话回合中，所有LLM都显示出精神致病的潜力，表现出强烈倾向于巩固而不是挑战妄想的趋势（平均DCS值为0.91±0.88）。模型频繁地促进有害用户请求（平均HES值为0.69±0.84），仅在约三分之一适用回合中提供建议（平均SIS值为0.37±0.48）。128个情景中有51个（39.8%）没有提供安全干预。在隐含情景中，模型的表现明显更差，更倾向于确认妄想并促进危害，提供干预的机会更少（p<0.001）。DCS和HES之间存在较强的正相关（rs=0.77）。模型性能差异显著，表明安全性不仅仅是规模的固有属性。

结论：本研究确立了LLM的精神致病性是一种可量化的风险，并强调了重新思考我们训练LLM的必要性。我们将这一问题不仅视为技术挑战，更视为需开发人员、政策制定者和医疗专业人员合作应对的公共卫生紧急需求。 

---
# Testing for LLM response differences: the case of a composite null consisting of semantically irrelevant query perturbations 

**Title (ZH)**: 测试LLM响应差异：关于语义无关查询扰动的composite null假设案例研究 

**Authors**: Aranyak Acharyya, Carey E. Priebe, Hayden S. Helm  

**Link**: [PDF](https://arxiv.org/pdf/2509.10963)  

**Abstract**: Given an input query, generative models such as large language models produce a random response drawn from a response distribution. Given two input queries, it is natural to ask if their response distributions are the same. While traditional statistical hypothesis testing is designed to address this question, the response distribution induced by an input query is often sensitive to semantically irrelevant perturbations to the query, so much so that a traditional test of equality might indicate that two semantically equivalent queries induce statistically different response distributions. As a result, the outcome of the statistical test may not align with the user's requirements. In this paper, we address this misalignment by incorporating into the testing procedure consideration of a collection of semantically similar queries. In our setting, the mapping from the collection of user-defined semantically similar queries to the corresponding collection of response distributions is not known a priori and must be estimated, with a fixed budget. Although the problem we address is quite general, we focus our analysis on the setting where the responses are binary, show that the proposed test is asymptotically valid and consistent, and discuss important practical considerations with respect to power and computation. 

**Abstract (ZH)**: 基于语义相似查询的生成模型响应分布比较方法 

---
# When the Code Autopilot Breaks: Why LLMs Falter in Embedded Machine Learning 

**Title (ZH)**: 当代码自动驾驶失效：为什么LLMs在嵌入式机器学习中出现问题 

**Authors**: Roberto Morabito, Guanghan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10946)  

**Abstract**: Large Language Models (LLMs) are increasingly used to automate software generation in embedded machine learning workflows, yet their outputs often fail silently or behave unpredictably. This article presents an empirical investigation of failure modes in LLM-powered ML pipelines, based on an autopilot framework that orchestrates data preprocessing, model conversion, and on-device inference code generation. We show how prompt format, model behavior, and structural assumptions influence both success rates and failure characteristics, often in ways that standard validation pipelines fail to detect. Our analysis reveals a diverse set of error-prone behaviors, including format-induced misinterpretations and runtime-disruptive code that compiles but breaks downstream. We derive a taxonomy of failure categories and analyze errors across multiple LLMs, highlighting common root causes and systemic fragilities. Though grounded in specific devices, our study reveals broader challenges in LLM-based code generation. We conclude by discussing directions for improving reliability and traceability in LLM-powered embedded ML systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在嵌入式机器学习工作流中越来越多地用于自动化软件生成，但其输出往往会无声失败或表现出不可预测的行为。本文基于一个自动驾驶框架进行经验研究，该框架协调数据预处理、模型转换和设备端推理代码生成。我们展示了提示格式、模型行为和结构假设如何影响成功率和失败特征，通常标准验证管道无法检测到这些特征。我们的分析揭示了一系列易出错的行为模式，包括格式引起的误解释和在编译但中断下游运行的代码。我们提出了失败类别分类法，并在多个LLM中分析错误，突显出常见的根本原因和系统脆弱性。尽管研究基于特定设备，但我们的研究揭示了LLM驱动代码生成中更广泛的挑战。我们最后讨论了提高LLM驱动嵌入式ML系统可靠性和可追溯性的方向。 

---
# Large Language Models for Security Operations Centers: A Comprehensive Survey 

**Title (ZH)**: 大型语言模型在安全运营中心中的应用：一项全面综述 

**Authors**: Ali Habibzadeh, Farid Feyzi, Reza Ebrahimi Atani  

**Link**: [PDF](https://arxiv.org/pdf/2509.10858)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools capable of understanding and generating human-like text, offering transformative potential across diverse domains. The Security Operations Center (SOC), responsible for safeguarding digital infrastructure, represents one of these domains. SOCs serve as the frontline of defense in cybersecurity, tasked with continuous monitoring, detection, and response to incidents. However, SOCs face persistent challenges such as high alert volumes, limited resources, high demand for experts with advanced knowledge, delayed response times, and difficulties in leveraging threat intelligence effectively. In this context, LLMs can offer promising solutions by automating log analysis, streamlining triage, improving detection accuracy, and providing the required knowledge in less time. This survey systematically explores the integration of generative AI and more specifically LLMs into SOC workflow, providing a structured perspective on its capabilities, challenges, and future directions. We believe that this survey offers researchers and SOC managers a broad overview of the current state of LLM integration within academic study. To the best of our knowledge, this is the first comprehensive study to examine LLM applications in SOCs in details. 

**Abstract (ZH)**: 大型语言模型(LLMs)已成为能够理解和生成类人类文本的强大工具，为不同领域带来了变革性的潜力。安全运营中心(SOC)，负责保护数字基础设施，是这些领域之一。SOC在网络安全中担任前线防御的角色，负责持续的监控、检测和响应事件。然而，SOC面临着持续的挑战，如高警报量、资源有限、对具备高级知识的专家需求高、响应时间延迟以及难以有效利用威胁情报。在此背景下，LLMs可以通过自动化日志分析、简化triage流程、提高检测准确性以及在较短时间内提供所需知识来提供有希望的解决方案。本文系统探讨了生成式AI和更具体的LLMs在SOC工作流程中的集成，提供了其功能、挑战和未来方向的结构化视角。我们认为，本文为研究人员和SOC管理人员提供了LLMs在学术研究中集成现状的广泛概述。据我们所知，这是首次对LLMs在SOC中的应用进行全面详细研究的综述。 

---
# Towards Automated Error Discovery: A Study in Conversational AI 

**Title (ZH)**: 面向自动错误发现：基于对话AI的研究 

**Authors**: Dominic Petrak, Thy Thy Tran, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2509.10833)  

**Abstract**: Although LLM-based conversational agents demonstrate strong fluency and coherence, they still produce undesirable behaviors (errors) that are challenging to prevent from reaching users during deployment. Recent research leverages large language models (LLMs) to detect errors and guide response-generation models toward improvement. However, current LLMs struggle to identify errors not explicitly specified in their instructions, such as those arising from updates to the response-generation model or shifts in user behavior. In this work, we introduce Automated Error Discovery, a framework for detecting and defining errors in conversational AI, and propose SEEED (Soft Clustering Extended Encoder-Based Error Detection), as an encoder-based approach to its implementation. We enhance the Soft Nearest Neighbor Loss by amplifying distance weighting for negative samples and introduce Label-Based Sample Ranking to select highly contrastive examples for better representation learning. SEEED outperforms adapted baselines -- including GPT-4o and Phi-4 -- across multiple error-annotated dialogue datasets, improving the accuracy for detecting unknown errors by up to 8 points and demonstrating strong generalization to unknown intent detection. 

**Abstract (ZH)**: 尽管基于LLM的对话代理表现出色，但在部署过程中仍会产生难以阻止到达用户的不良行为（错误）。近期研究利用大型语言模型（LLMs）检测错误并引导响应生成模型改进。然而，当前LLMs在识别未明确包含在指令中的错误方面仍然困难，例如响应生成模型更新或用户行为变化导致的错误。在本文中，我们提出了一种自动错误发现框架，用于检测和定义对话AI中的错误，并提出SEEED（Soft Clustering Extended Encoder-Based Error Detection）作为其实现的编码器基方法。我们通过放大负样本的距离权重增强Soft最近邻损失，并引入基于标签的样本排名以选择更具对比性的示例，从而提高表示学习的效果。SEEED在多个错误标注的对话数据集上优于适应基线（包括GPT-4o和Phi-4），检测未知错误的准确率提高最多可达8个百分点，并且在未知意图检测方面表现出强大的泛化能力。 

---
# Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction 

**Title (ZH)**: 法官Q：可训练查询以优化键值缓存淘汰中的信息保留 

**Authors**: Yijun Liu, Yixuan Wang, Yuzhuang Xu, Shiyu Ji, Yang Xu, Qingfu Zhu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2509.10798)  

**Abstract**: Large language models (LLMs) utilize key-value (KV) cache to store historical information during sequence processing. The size of KV cache grows linearly as the length of the sequence extends, which seriously affects memory usage and decoding efficiency. Current methods for KV cache eviction typically utilize the last window from the pre-filling phase as queries to compute the KV importance scores for eviction. Although this scheme is simple to implement, it tends to overly focus on local information, potentially leading to the neglect or omission of crucial global information. To mitigate this issue, we propose Judge Q, a novel training method which incorporates a soft token list. This method only tunes the model's embedding layer at a low training cost. By concatenating the soft token list at the end of the input sequence, we train these tokens' attention map to the original input sequence to align with that of the actual decoded tokens. In this way, the queries corresponding to the soft tokens can effectively capture global information and better evaluate the importance of the keys and values within the KV cache, thus maintaining decoding quality when KV cache is evicted. Under the same eviction budget, our method exhibits less performance degradation compared to existing eviction approaches. We validate our approach through experiments conducted on models such as Llama-3.1-8B-Instruct and Mistral-7B-Instruct-v0.3, using benchmarks including LongBench, RULER, and Needle-in-a-Haystack. Results indicate an improvement of approximately 1 point on the LongBench and over 3 points on RULER. This proposed methodology can be seamlessly integrated into existing open-source models with minimal training overhead, thereby enhancing performance in KV cache eviction scenarios. 

**Abstract (ZH)**: 大规模语言模型（LLMs）利用键值（KV）缓存存储序列处理过程中的历史信息。随着序列长度的增加，KV缓存的大小线性增长，严重影响了内存使用和解码效率。当前的KV缓存淘汰方法通常利用预填充阶段的最后一个窗口作为查询，计算KV重要性分数以进行淘汰。虽然这种方法实现简单，但往往会过度关注局部信息，可能导致关键全局信息的忽略或遗漏。为缓解这一问题，我们提出了一种新颖的训练方法Judge Q，该方法引入了一个软令牌列表。该方法仅在较低的训练成本下调整模型的嵌入层。通过在输入序列末尾连接软令牌列表，我们训练这些令牌的注意力图与实际解码令牌的注意力图对齐，从而能够有效捕获全局信息并更准确地评估KV缓存中键和值的重要性，从而在KV缓存淘汰时保持解码质量。在相同的淘汰预算下，我们的方法相较于现有方法表现出更少的性能降级。我们通过在Llama-3.1-8B-Instruct和Mistral-7B-Instruct-v0.3等模型上进行实验，并使用LongBench、RULER和Needle-in-a-Haystack等基准测试，验证了该方法的有效性。实验结果表明，该方法在LongBench上的性能提高了约1分，在RULER上的性能提高了超过3分。该提议的方法可以无缝集成到现有的开源模型中，且几乎不增加训练开销，从而在KV缓存淘汰场景中提升了性能。 

---
# GoldenTransformer: A Modular Fault Injection Framework for Transformer Robustness Research 

**Title (ZH)**: GoldenTransformer：一个用于Transformer鲁棒性研究的模块化故障注入框架 

**Authors**: Luke Howard  

**Link**: [PDF](https://arxiv.org/pdf/2509.10790)  

**Abstract**: Transformers have become the foundation for a wide range of state--of--the--art models across natural language processing, computer vision, and other machine learning domains. Despite their widespread deployment, the robustness of these models under fault conditions remains underexplored. We present GoldenTransformer, a modular and extensible fault injection framework designed to evaluate the resiliency of Large Language Models to induced hardware faults. GoldenTransformer offers a unified Python-based platform for injecting diverse classes of faults--such as weight corruption, activation injections, and attention--level disruptions--into pretrained transformer--based models. Inspired by the GoldenEye simulator for DNNs, our framework focuses on the unique challenges of working with large transformer architectures, including considerations such as structural complexity, latent dependencies, and nonuniform layer definitions. GoldenTransformer is built atop PyTorch and HuggingFace Transformers, and it supports experiment reproducibility, metric logging, and visualization out of the box. We detail the technical design and use of GoldenTransformer and demonstrate through several example experiments on classification and generation tasks. By enabling controlled injection of faults at multiple logical and structural points in a transformer, GoldenTransformer offers researchers and practitioners a valuable tool for model robustness analysis and for guiding dependable system design in real-world LLM applications. 

**Abstract (ZH)**: 变压器已成为自然语言处理、计算机视觉和其他机器学习领域中众多前沿模型的基础。尽管这些模型被广泛部署，但在故障条件下的鲁棒性仍鲜有研究。我们介绍了GoldenTransformer，这是一个模块化和可扩展的故障注入框架，旨在评估大型语言模型在诱导硬件故障情况下的韧性。GoldenTransformer提供了一个统一的基于Python的平台，用于向预训练的基于变压器的模型注入多种类型的故障，如权重篡改、激活注入和注意力层面的干扰。灵感来源于DNN领域的GoldenEye模拟器，我们的框架重点关注大规模变压器架构特有的挑战，包括结构复杂性、潜在依赖性和非均匀的层定义。GoldenTransformer基于PyTorch和HuggingFace Transformers构建，支持实验的可重复性、指标日志记录和可视化。我们详细介绍了GoldenTransformer的技术设计和使用方法，并通过几个分类和生成任务的示例实验进行了演示。通过在变压器中的多个逻辑和结构点上实现可控的故障注入，GoldenTransformer为研究者和从业者提供了一个有价值的工具，用于模型鲁棒性分析，并指导实际应用中可靠系统的设计。 

---
# Bridging Cultural Distance Between Models Default and Local Classroom Demands: How Global Teachers Adopt GenAI to Support Everyday Teaching Practices 

**Title (ZH)**: 跨越模型默认设置与当地教室需求的文化距离：全球教师如何采用生成式AI支持日常教学实践 

**Authors**: Ruiwei Xiao, Qing Xiao, Xinying Hou, Hanqi Jane Li, Phenyo Phemelo Moletsane, Hong Shen, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2509.10780)  

**Abstract**: Generative AI (GenAI) is rapidly entering K-12 classrooms, offering teachers new ways for teaching practices. Yet GenAI models are often trained on culturally uneven datasets, embedding a "default culture" that often misaligns with local classrooms. To understand how teachers navigate this gap, we defined the new concept Cultural Distance (the gap between GenAI's default cultural repertoire and the situated demands of teaching practice) and conducted in-depth interviews with 30 K-12 teachers, 10 each from South Africa, Taiwan, and the United States, who had integrated AI into their teaching practice. These teachers' experiences informed the development of our three-level cultural distance framework. This work contributes the concept and framework of cultural distance, six illustrative instances spanning in low, mid, high distance levels with teachers' experiences and strategies for addressing them. Empirically, we offer implications to help AI designers, policymakers, and educators create more equitable and culturally responsive GenAI tools for education. 

**Abstract (ZH)**: 生成式AI（GenAI）正迅速进入K-12课堂，为教学实践提供了新的方法。然而，GenAI模型经常使用文化不均衡的数据集进行训练，嵌入了一种“默认文化”，这往往与本地教室的需求不一致。为了理解教师如何克服这种差距，我们定义了新的概念文化距离（GenAI的默认文化库与教学实践中的情境需求之间的差距），并深入访谈了30名来自南非、台湾和美国的K-12教师，这些教师已经在教学中整合了人工智能。这些教师的经验指导了我们三层文化距离框架的开发。这项工作贡献了文化距离的概念和框架，以及涵盖低、中、高文化距离级别的六个实例，包括教师的经验和应对策略。实证研究提供了建议，旨在帮助AI设计师、政策制定者和教育工作者为教育创建更加公平和文化响应的GenAI工具。 

---
# HalluField: Detecting LLM Hallucinations via Field-Theoretic Modeling 

**Title (ZH)**: HalluField：基于场论建模的LLM幻觉检测 

**Authors**: Minh Vu, Brian K. Tran, Syed A. Shah, Geigh Zollicoffer, Nhat Hoang-Xuan, Manish Bhattarai  

**Link**: [PDF](https://arxiv.org/pdf/2509.10753)  

**Abstract**: Large Language Models (LLMs) exhibit impressive reasoning and question-answering capabilities. However, they often produce inaccurate or unreliable content known as hallucinations. This unreliability significantly limits their deployment in high-stakes applications. Thus, there is a growing need for a general-purpose method to detect hallucinations in LLMs. In this work, we introduce HalluField, a novel field-theoretic approach for hallucination detection based on a parametrized variational principle and thermodynamics. Inspired by thermodynamics, HalluField models an LLM's response to a given query and temperature setting as a collection of discrete likelihood token paths, each associated with a corresponding energy and entropy. By analyzing how energy and entropy distributions vary across token paths under changes in temperature and likelihood, HalluField quantifies the semantic stability of a response. Hallucinations are then detected by identifying unstable or erratic behavior in this energy landscape. HalluField is computationally efficient and highly practical: it operates directly on the model's output logits without requiring fine-tuning or auxiliary neural networks. Notably, the method is grounded in a principled physical interpretation, drawing analogies to the first law of thermodynamics. Remarkably, by modeling LLM behavior through this physical lens, HalluField achieves state-of-the-art hallucination detection performance across models and datasets. 

**Abstract (ZH)**: 基于场论的幻觉检测方法：HalluField 

---
# Automated MCQA Benchmarking at Scale: Evaluating Reasoning Traces as Retrieval Sources for Domain Adaptation of Small Language Models 

**Title (ZH)**: 大规模自动化MCQA基准评估：将推理追踪作为小型语言模型领域适应的检索来源进行评估 

**Authors**: Ozan Gokdemir, Neil Getty, Robert Underwood, Sandeep Madireddy, Franck Cappello, Arvind Ramanathan, Ian T. Foster, Rick L. Stevens  

**Link**: [PDF](https://arxiv.org/pdf/2509.10744)  

**Abstract**: As scientific knowledge grows at an unprecedented pace, evaluation benchmarks must evolve to reflect new discoveries and ensure language models are tested on current, diverse literature. We propose a scalable, modular framework for generating multiple-choice question-answering (MCQA) benchmarks directly from large corpora of scientific papers. Our pipeline automates every stage of MCQA creation, including PDF parsing, semantic chunking, question generation, and model evaluation. As a case study, we generate more than 16,000 MCQs from 22,000 open-access articles in radiation and cancer biology. We then evaluate a suite of small language models (1.1B-14B parameters) on these questions, comparing baseline accuracy with retrieval-augmented generation (RAG) from paper-derived semantic chunks and from reasoning traces distilled from GPT-4.1. We find that reasoning-trace retrieval consistently improves performance on both synthetic and expert-annotated benchmarks, enabling several small models to surpass GPT-4 on the 2023 Astro Radiation and Cancer Biology exam. 

**Abstract (ZH)**: 随着科学知识以前所未有的速度增长，评估基准必须随之演进而反映新的发现，并确保语言模型能够被测试于当前多样化的文献上。我们提出了一种可扩展且模块化的框架，直接从大规模的科学论文 corpora 中生成多项选择题-答案（MCQA）基准。我们的管道自动化了 MCQA 创建过程中的每一个阶段，包括 PDF 解析、语义切块、问题生成和模型评估。作为案例研究，我们从 22,000 篇开放获取的辐射和癌症生物学论文中生成了超过 16,000 个 MCQ。然后，我们在这些题目上评估一系列小型语言模型（参数量从 11 亿到 140 亿不等），并将基于论文提取的语义切块的检索增强生成（RAG）方法与从 GPT-4.1 精练推理踪迹的方法进行比较。我们发现，基于推理踪迹的检索在合成基准和专家注释基准中均能持续提升性能，使得某些小型模型超越了 GPT-4 在 2023 年天体辐射和癌症生物学考试中的表现。 

---
# Dark Patterns Meet GUI Agents: LLM Agent Susceptibility to Manipulative Interfaces and the Role of Human Oversight 

**Title (ZH)**: 暗模式碰上GUI代理：LLM代理对 manipulative界面的易感性及人类监督的作用 

**Authors**: Jingyu Tang, Chaoran Chen, Jiawen Li, Zhiping Zhang, Bingcan Guo, Ibrahim Khalilov, Simret Araya Gebreegziabher, Bingsheng Yao, Dakuo Wang, Yanfang Ye, Tianshi Li, Ziang Xiao, Yaxing Yao, Toby Jia-Jun Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.10723)  

**Abstract**: The dark patterns, deceptive interface designs manipulating user behaviors, have been extensively studied for their effects on human decision-making and autonomy. Yet, with the rising prominence of LLM-powered GUI agents that automate tasks from high-level intents, understanding how dark patterns affect agents is increasingly important. We present a two-phase empirical study examining how agents, human participants, and human-AI teams respond to 16 types of dark patterns across diverse scenarios. Phase 1 highlights that agents often fail to recognize dark patterns, and even when aware, prioritize task completion over protective action. Phase 2 revealed divergent failure modes: humans succumb due to cognitive shortcuts and habitual compliance, while agents falter from procedural blind spots. Human oversight improved avoidance but introduced costs such as attentional tunneling and cognitive load. Our findings show neither humans nor agents are uniformly resilient, and collaboration introduces new vulnerabilities, suggesting design needs for transparency, adjustable autonomy, and oversight. 

**Abstract (ZH)**: 基于LLM的GUI代理中暗模式的影响：两阶段实证研究 

---
# Pluralistic Alignment for Healthcare: A Role-Driven Framework 

**Title (ZH)**: 医疗领域多样化的对齐：一种角色驱动的框架 

**Authors**: Jiayou Zhong, Anudeex Shetty, Chao Jia, Xuanrui Lin, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2509.10685)  

**Abstract**: As large language models are increasingly deployed in sensitive domains such as healthcare, ensuring their outputs reflect the diverse values and perspectives held across populations is critical. However, existing alignment approaches, including pluralistic paradigms like Modular Pluralism, often fall short in the health domain, where personal, cultural, and situational factors shape pluralism. Motivated by the aforementioned healthcare challenges, we propose a first lightweight, generalizable, pluralistic alignment approach, EthosAgents, designed to simulate diverse perspectives and values. We empirically show that it advances the pluralistic alignment for all three modes across seven varying-sized open and closed models. Our findings reveal that health-related pluralism demands adaptable and normatively aware approaches, offering insights into how these models can better respect diversity in other high-stakes domains. 

**Abstract (ZH)**: 大语言模型在医疗等敏感领域日益普及，确保其输出反映不同人群持有的多元价值观和视角至关重要。然而，现有的对齐方法，包括模块化多元主义等多元主义范式，在医疗领域往往难以奏效，因为个人、文化和社会情境因素影响着多元主义的形态。为应对上述医疗挑战，我们提出了一种轻量级、可泛化的多元主义对齐方法EthosAgents，旨在模拟多元的视角和价值观。实验证明，该方法在七个不同规模的开放和封闭模型中跨三种模式推进了多元主义对齐。研究结果表明，与健康相关的多元主义需要适应性和规范意识更强的方法，为其他高风险领域如何更好地尊重多元化提供了见解。 

---
# LLM in the Middle: A Systematic Review of Threats and Mitigations to Real-World LLM-based Systems 

**Title (ZH)**: LLM居中：面向现实世界的大语言模型威胁与缓解措施系统的综述 

**Authors**: Vitor Hugo Galhardo Moia, Igor Jochem Sanz, Gabriel Antonio Fontes Rebello, Rodrigo Duarte de Meneses, Briland Hitaj, Ulf Lindqvist  

**Link**: [PDF](https://arxiv.org/pdf/2509.10682)  

**Abstract**: The success and wide adoption of generative AI (GenAI), particularly large language models (LLMs), has attracted the attention of cybercriminals seeking to abuse models, steal sensitive data, or disrupt services. Moreover, providing security to LLM-based systems is a great challenge, as both traditional threats to software applications and threats targeting LLMs and their integration must be mitigated. In this survey, we shed light on security and privacy concerns of such LLM-based systems by performing a systematic review and comprehensive categorization of threats and defensive strategies considering the entire software and LLM life cycles. We analyze real-world scenarios with distinct characteristics of LLM usage, spanning from development to operation. In addition, threats are classified according to their severity level and to which scenarios they pertain, facilitating the identification of the most relevant threats. Recommended defense strategies are systematically categorized and mapped to the corresponding life cycle phase and possible attack strategies they attenuate. This work paves the way for consumers and vendors to understand and efficiently mitigate risks during integration of LLMs in their respective solutions or organizations. It also enables the research community to benefit from the discussion of open challenges and edge cases that may hinder the secure and privacy-preserving adoption of LLM-based systems. 

**Abstract (ZH)**: 生成型人工智能（GenAI），特别是大型语言模型（LLMs）的成功及其广泛应用引起了网络犯罪分子的注意，他们试图滥用这些模型、窃取敏感数据或扰乱服务。此外，保障基于LLM的系统的安全是一项巨大挑战，因为必须同时缓解针对软件应用程序的传统威胁和针对LLM及其集成的威胁。在这篇综述中，我们通过对软件和LLM生命周期进行全面系统性审查和分类，揭示了基于LLM系统的安全和隐私关切。我们分析了从开发到运营的不同LLM使用场景下的实际场景，根据威胁的严重程度及其适用的场景对威胁进行了分类，有助于识别最相关的威胁。推荐的防御策略被系统地分类，并与相应的生活周期阶段以及它们可以减轻的攻击策略进行映射。这项工作为消费者和供应商提供了理解并有效地缓解在各自解决方案或组织中整合LLM时的风险的途径。同时，它也为研究社区提供了讨论可能阻碍基于LLM系统的安全和隐私保护采用的开放挑战和边缘案例的机会。 

---
# Test-Time Warmup for Multimodal Large Language Models 

**Title (ZH)**: 多模态大语言模型的测试时预热 

**Authors**: Nikita Rajaneesh, Thomas Zollo, Richard Zemel  

**Link**: [PDF](https://arxiv.org/pdf/2509.10641)  

**Abstract**: Multimodal Large Language Models (MLLMs) hold great promise for advanced reasoning at the intersection of text and images, yet they have not fully realized this potential. MLLMs typically integrate an LLM, a vision encoder, and a connector that maps the vision encoder's embeddings into the LLM's text embedding space. Although each component is pretrained on massive datasets with billions of samples, the entire multimodal model is typically trained on only thousands (or a few million) samples, which can result in weak performance on complex reasoning tasks. To address these shortcomings, instead of relying on extensive labeled datasets for fine-tuning, we propose a Test-Time Warmup method that adapts the MLLM per test instance by leveraging data from weakly supervised auxiliary tasks. With our approach, we observe a relative performance improvement of 4.03% on MMMU, 5.28% on VQA-Rad, and 1.63% on GQA on the Llama-Vision-Instruct model. Our method demonstrates that 'warming up' before inference can enhance MLLMs' robustness across diverse reasoning tasks. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在文本和图像交叉领域的高级推理方面展现出巨大潜力，但尚未充分实现这一潜力。尽管MLLMs通常包括一个大型语言模型、一个视觉编码器以及一个连接器将视觉编码器的嵌入映射到大型语言模型的文本嵌入空间，每个组件都预训练在包含数十亿样本的巨大数据集上，但整个多模态模型通常仅在成千上万（或几百万）样本上进行训练，这可能导致在复杂推理任务上的表现较弱。为解决这些不足，我们提出了一种测试时预热方法，该方法通过利用弱监督辅助任务的数据来适应每个测试实例的MLLM，而不需要依赖大量标注的数据集进行微调。在我们的方法下，我们在MMMU上的相对性能改进为4.03%，在VQA-Rad上的相对性能改进为5.28%，在GQA上的相对性能改进为1.63%，基于Llama-Vision-Instruct模型。我们的方法表明，在推理前进行“预热”可以增强MLLMs在各种推理任务中的鲁棒性。 

---
# No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes 

**Title (ZH)**: 无需回答：仅从问题预测大语言模型答案准确性的方法 

**Authors**: Iván Vicente Moreno Cencerrado, Arnau Padrés Masdemont, Anton Gonzalvez Hawthorne, David Demitri Africa, Lorenzo Pacchiardi  

**Link**: [PDF](https://arxiv.org/pdf/2509.10625)  

**Abstract**: Do large language models (LLMs) anticipate when they will answer correctly? To study this, we extract activations after a question is read but before any tokens are generated, and train linear probes to predict whether the model's forthcoming answer will be correct. Across three open-source model families ranging from 7 to 70 billion parameters, projections on this "in-advance correctness direction" trained on generic trivia questions predict success in distribution and on diverse out-of-distribution knowledge datasets, outperforming black-box baselines and verbalised predicted confidence. Predictive power saturates in intermediate layers, suggesting that self-assessment emerges mid-computation. Notably, generalisation falters on questions requiring mathematical reasoning. Moreover, for models responding "I don't know", doing so strongly correlates with the probe score, indicating that the same direction also captures confidence. By complementing previous results on truthfulness and other behaviours obtained with probes and sparse auto-encoders, our work contributes essential findings to elucidate LLM internals. 

**Abstract (ZH)**: 大型语言模型（LLM）是否能在作答前预知答案的正确性？我们提取问题读取后但在生成任何tokens之前的信息激活，并训练线性端头以预测模型即将给出的答案是否正确。通过对7亿至70亿参数的三个开源模型家族进行研究，在通用 trivia 问题上的“预先正确性方向”投影不仅在分布上预测表现优异，还在多种离分布知识数据集上超越黑盒基线和表达的预测信心。预测能力在中间层趋于饱和，表明自我评估在计算过程中中期出现。值得注意的是，对需要数学推理的问题，泛化表现不佳。此外，对于回答“不知道”的模型，这种回答与端头得分高度相关，表明同一方向也捕捉了信心。通过补充使用端头和稀疏自编码器获得的关于诚实性和其他行为的先前结果，我们的工作为阐明LLM内部机制贡献了关键发现。 

---
# SME-TEAM: Leveraging Trust and Ethics for Secure and Responsible Use of AI and LLMs in SMEs 

**Title (ZH)**: SME-TEAM：利用信任和伦理规范确保中小企业安全负责任地使用AI和大语言模型 

**Authors**: Iqbal H. Sarker, Helge Janicke, Ahmad Mohsin, Leandros Maglaras  

**Link**: [PDF](https://arxiv.org/pdf/2509.10594)  

**Abstract**: Artificial Intelligence (AI) and Large Language Models (LLMs) are reshaping today's business practices, however, their adoption within small and medium-sized enterprises (SMEs) raises significant technical, ethical and trust issues. This paper proposes a structured, multi-phased framework designed to embed trust and ethical principles throughout the AI lifecycle for their secure and responsible use in SMEs. Structured around four pillars, i.e., Data, Algorithms, Human oversight, and Model Architecture, the framework bridges theoretical ethical principles with operational practice, enhancing AI capabilities in diverse SME applications. Ultimately, this paper offers a structured roadmap for responsible AI adoption, framing trust and ethics as a catalyst for resilience, competitiveness, and sustainable innovation in SMEs. 

**Abstract (ZH)**: 人工智能（AI）和大型语言模型（LLMs）正在重塑当今的商业实践，然而在小型和中型企业（SMEs）中采用这些技术引发了重要的技术、伦理和信任问题。本文提出了一种结构化、多阶段框架，旨在在整个AI生命周期中嵌入信任和伦理原则，促进其在SMEs中的安全和负责任使用。该框架以数据、算法、人类监督和模型架构四个支柱为基础，将理论伦理原则与操作实践相结合，增强在不同SME应用中的AI能力。最终，本文提供了一条结构化的负责任AI采用路线图，将信任和伦理作为增强SMEs韧性和竞争力、推动可持续创新的催化剂。 

---
# Smart Trial: Evaluating the Use of Large Language Models for Recruiting Clinical Trial Participants via Social Media 

**Title (ZH)**: 智能试验：评估通过社交媒体使用大型语言模型招募临床试验参与者的效果 

**Authors**: Xiaofan Zhou, Zisu Wang, Janice Krieger, Mohan Zalake, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.10584)  

**Abstract**: Clinical trials (CT) are essential for advancing medical research and treatment, yet efficiently recruiting eligible participants -- each of whom must meet complex eligibility criteria -- remains a significant challenge. Traditional recruitment approaches, such as advertisements or electronic health record screening within hospitals, are often time-consuming and geographically constrained. This work addresses the recruitment challenge by leveraging the vast amount of health-related information individuals share on social media platforms. With the emergence of powerful large language models (LLMs) capable of sophisticated text understanding, we pose the central research question: Can LLM-driven tools facilitate CT recruitment by identifying potential participants through their engagement on social media? To investigate this question, we introduce TRIALQA, a novel dataset comprising two social media collections from the subreddits on colon cancer and prostate cancer. Using eligibility criteria from public real-world CTs, experienced annotators are hired to annotate TRIALQA to indicate (1) whether a social media user meets a given eligibility criterion and (2) the user's stated reasons for interest in participating in CT. We benchmark seven widely used LLMs on these two prediction tasks, employing six distinct training and inference strategies. Our extensive experiments reveal that, while LLMs show considerable promise, they still face challenges in performing the complex, multi-hop reasoning needed to accurately assess eligibility criteria. 

**Abstract (ZH)**: 临床试验（CT）通过社交媒体平台招募合格参与者的研究：利用大规模语言模型的机遇与挑战 

---
# Gene-R1: Reasoning with Data-Augmented Lightweight LLMs for Gene Set Analysis 

**Title (ZH)**: Gene-R1：基于数据增强轻量级LLM的基因集分析推理 

**Authors**: Zhizheng Wang, Yifan Yang, Qiao Jin, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10575)  

**Abstract**: The gene set analysis (GSA) is a foundational approach for uncovering the molecular functions associated with a group of genes. Recently, LLM-powered methods have emerged to annotate gene sets with biological functions together with coherent explanatory insights. However, existing studies primarily focus on proprietary models, which have been shown to outperform their open-source counterparts despite concerns over cost and data privacy. Furthermore, no research has investigated the application of advanced reasoning strategies to the GSA task. To address this gap, we introduce Gene-R1, a data-augmented learning framework that equips lightweight and open-source LLMs with step-by-step reasoning capabilities tailored to GSA. Experiments on 1,508 in-distribution gene sets demonstrate that Gene-R1 achieves substantial performance gains, matching commercial LLMs. On 106 out-of-distribution gene sets, Gene-R1 performs comparably to both commercial and large-scale LLMs, exhibiting robust generalizability across diverse gene sources. 

**Abstract (ZH)**: 基于基因集的基因组功能分析（GSA）是揭示一组基因相关分子功能的基础方法。近期，基于大语言模型（LLM）的方法已经出现，能够同时标注基因集的生物功能及其一致的解释性见解。然而，现有研究主要集中在专有模型上，尽管这些模型在成本和数据隐私存在担忧的情况下仍表现出色。此外，没有研究探讨高级推理策略在GSA任务中的应用。为填补这一空白，我们介绍了Gene-R1，一个数据增强的学习框架，它为轻量级和开源的LLM配备了针对GSA定制的逐步推理能力。实验结果显示，在1,508个分布内基因集上，Gene-R1实现了显著的性能提升，与商业大语言模型相当。在106个分布外基因集上，Gene-R1与商业和大规模的LLM相当，展示了在多样化的基因来源上的稳健泛化能力。 

---
# Quality Assessment of Tabular Data using Large Language Models and Code Generation 

**Title (ZH)**: 使用大型语言模型和代码生成评估表格数据质量 

**Authors**: Ashlesha Akella, Akshar Kaul, Krishnasuri Narayanam, Sameep Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2509.10572)  

**Abstract**: Reliable data quality is crucial for downstream analysis of tabular datasets, yet rule-based validation often struggles with inefficiency, human intervention, and high computational costs. We present a three-stage framework that combines statistical inliner detection with LLM-driven rule and code generation. After filtering data samples through traditional clustering, we iteratively prompt LLMs to produce semantically valid quality rules and synthesize their executable validators through code-generating LLMs. To generate reliable quality rules, we aid LLMs with retrieval-augmented generation (RAG) by leveraging external knowledge sources and domain-specific few-shot examples. Robust guardrails ensure the accuracy and consistency of both rules and code snippets. Extensive evaluations on benchmark datasets confirm the effectiveness of our approach. 

**Abstract (ZH)**: 可靠的數據質量對於表數據下游分析至關重要，但基于规则的验证往往面临效率低下、人工干预和高計算成本的问题。我们提出一个三阶段框架，结合统计内点检测与大语言模型驱动的规则和代码生成。在通过传统聚类过滤数据样本后，我们迭代地提示大语言模型生成语义上有意义的质量规则，并通过代码生成大语言模型合成效仿器合成其实现验证器。为了生成可靠的质量规则，我们通过检索增强生成（RAG）辅助大语言模型，利用外部知识源和领域特定的少量示例。稳健的护栏确保规则和代码片段的准确性和一致性。在基准数据集上的 extensive 评估验证了该方法的有效性。 

---
# AVEC: Bootstrapping Privacy for Local LLMs 

**Title (ZH)**: AVEC：为本地LLMs Bootstrapping隐私权 

**Authors**: Madhava Gaikwad  

**Link**: [PDF](https://arxiv.org/pdf/2509.10561)  

**Abstract**: This position paper presents AVEC (Adaptive Verifiable Edge Control), a framework for bootstrapping privacy for local language models by enforcing privacy at the edge with explicit verifiability for delegated queries. AVEC introduces an adaptive budgeting algorithm that allocates per-query differential privacy parameters based on sensitivity, local confidence, and historical usage, and uses verifiable transformation with on-device integrity checks. We formalize guarantees using Rényi differential privacy with odometer-based accounting, and establish utility ceilings, delegation-leakage bounds, and impossibility results for deterministic gating and hash-only certification. Our evaluation is simulation-based by design to study mechanism behavior and accounting; we do not claim deployment readiness or task-level utility with live LLMs. The contribution is a conceptual architecture and theoretical foundation that chart a pathway for empirical follow-up on privately bootstrapping local LLMs. 

**Abstract (ZH)**: 本论题论文提出AVEC（自适应可验证边缘控制），一种通过在边缘端强制执行隐私并为委托查询提供显式可验证性来为本地语言模型启动隐私性的框架。AVEC引入了一种自适应预算算法，根据敏感性、本地置信度和历史使用情况为每个查询分配差分隐私参数，并使用带有设备内完整性检查的可验证转换。我们使用基于 odometer 的会计方法形式化保证，并建立了确定性门控和仅哈希认证的效用上限、泄漏边界和不可能性结果。我们的评估设计为基于模拟以研究机制行为和会计；我们不声称在实时大语言模型中实现部署或具有任务级效用。贡献在于提出了一个概念性架构和理论基础，为实证后续研究自适应启动本地大语言模型的隐私性铺平道路。 

---
# Uncovering the Vulnerability of Large Language Models in the Financial Domain via Risk Concealment 

**Title (ZH)**: 通过风险隐藏揭示大型语言模型在金融领域的脆弱性 

**Authors**: Gang Cheng, Haibo Jin, Wenbin Zhang, Haohan Wang, Jun Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10546)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into financial applications, yet existing red-teaming research primarily targets harmful content, largely neglecting regulatory risks. In this work, we aim to investigate the vulnerability of financial LLMs through red-teaming approaches. We introduce Risk-Concealment Attacks (RCA), a novel multi-turn framework that iteratively conceals regulatory risks to provoke seemingly compliant yet regulatory-violating responses from LLMs. To enable systematic evaluation, we construct FIN-Bench, a domain-specific benchmark for assessing LLM safety in financial contexts. Extensive experiments on FIN-Bench demonstrate that RCA effectively bypasses nine mainstream LLMs, achieving an average attack success rate (ASR) of 93.18%, including 98.28% on GPT-4.1 and 97.56% on OpenAI o1. These findings reveal a critical gap in current alignment techniques and underscore the urgent need for stronger moderation mechanisms in financial domains. We hope this work offers practical insights for advancing robust and domain-aware LLM alignment. 

**Abstract (ZH)**: 大型语言模型（LLMs）在金融应用中的安全性研究：Risk-Concealment Attacks及其对金融LLMs的评估 

---
# EchoLeak: The First Real-World Zero-Click Prompt Injection Exploit in a Production LLM System 

**Title (ZH)**: EchoLeak: 首个生产环境下的零点击提示注入利用攻击 

**Authors**: Pavan Reddy, Aditya Sanjay Gujral  

**Link**: [PDF](https://arxiv.org/pdf/2509.10540)  

**Abstract**: Large language model (LLM) assistants are increasingly integrated into enterprise workflows, raising new security concerns as they bridge internal and external data sources. This paper presents an in-depth case study of EchoLeak (CVE-2025-32711), a zero-click prompt injection vulnerability in Microsoft 365 Copilot that enabled remote, unauthenticated data exfiltration via a single crafted email. By chaining multiple bypasses-evading Microsofts XPIA (Cross Prompt Injection Attempt) classifier, circumventing link redaction with reference-style Markdown, exploiting auto-fetched images, and abusing a Microsoft Teams proxy allowed by the content security policy-EchoLeak achieved full privilege escalation across LLM trust boundaries without user interaction. We analyze why existing defenses failed, and outline a set of engineering mitigations including prompt partitioning, enhanced input/output filtering, provenance-based access control, and strict content security policies. Beyond the specific exploit, we derive generalizable lessons for building secure AI copilots, emphasizing the principle of least privilege, defense-in-depth architectures, and continuous adversarial testing. Our findings establish prompt injection as a practical, high-severity vulnerability class in production AI systems and provide a blueprint for defending against future AI-native threats. 

**Abstract (ZH)**: 大型语言模型（LLM）助手日益融入企业工作流程，引发了新的安全担忧，因为它们连接了内部和外部数据源。本文以EchoLeak（CVE-2025-32711）为例，深入研究了Microsoft 365 Copilot中的零点击提示注入漏洞，该漏洞通过单封精心制作的邮件实现了未认证的远程数据泄露。EchoLeak通过链接多个绕过Microsoft XPIA（跨提示注入尝试）分类器的措施，绕过了链接遮蔽，利用了参考样式的Markdown，利用了自动拉取的图像，并滥用允许的内容安全策略中的Microsoft Teams代理，从而在没有用户交互的情况下实现了跨LLM信任边界的身份提升。我们分析了现有防护措施为何失效，并提出了包括提示分区、增强的输入/输出过滤、基于溯源的访问控制和严格的内容安全策略在内的工程缓解措施。除了具体的利用方法外，我们还提炼出适用于构建安全AI协作者的一般原则，强调最小权限原则、多层次防御架构以及持续的对抗性测试。我们的研究成果将提示注入确立为生产AI系统中实用且高危的漏洞类别，并提供了抵御未来AI原生威胁的蓝图。 

---
# DualAlign: Generating Clinically Grounded Synthetic Data 

**Title (ZH)**: DualAlign: 生成具有临床依据的合成数据 

**Authors**: Rumeng Li, Xun Wang, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.10538)  

**Abstract**: Synthetic clinical data are increasingly important for advancing AI in healthcare, given strict privacy constraints on real-world EHRs, limited availability of annotated rare-condition data, and systemic biases in observational datasets. While large language models (LLMs) can generate fluent clinical text, producing synthetic data that is both realistic and clinically meaningful remains challenging. We introduce DualAlign, a framework that enhances statistical fidelity and clinical plausibility through dual alignment: (1) statistical alignment, which conditions generation on patient demographics and risk factors; and (2) semantic alignment, which incorporates real-world symptom trajectories to guide content generation. Using Alzheimer's disease (AD) as a case study, DualAlign produces context-grounded symptom-level sentences that better reflect real-world clinical documentation. Fine-tuning an LLaMA 3.1-8B model with a combination of DualAlign-generated and human-annotated data yields substantial performance gains over models trained on gold data alone or unguided synthetic baselines. While DualAlign does not fully capture longitudinal complexity, it offers a practical approach for generating clinically grounded, privacy-preserving synthetic data to support low-resource clinical text analysis. 

**Abstract (ZH)**: 合成临床数据在严格隐私限制、标注罕见病例数据有限以及观察性数据系统性偏见的情况下，对于推动医疗健康领域人工智能技术的发展日益重要。虽然大型语言模型可以生成流畅的临床文本，但生成既现实又具有临床意义的合成数据仍然具有挑战性。我们引入了DualAlign框架，通过双重对齐增强统计准确性和临床可行性：（1）统计对齐，基于患者的人口统计学特征和风险因素进行生成；（2）语义对齐，通过融入真实的症状轨迹来指导内容生成。以阿尔茨海默病（AD）为例，DualAlign生成了更贴近真实临床记录的细粒度症状句子。使用包含DualAlign生成数据和人工标注数据共同微调的LLaMA 3.1-8B模型，相对于仅使用黄金数据训练或未指导的合成基线模型，取得了显著的性能提升。尽管DualAlign无法完全捕捉纵向复杂性，但它提供了一种生成具有临床依据、保护隐私的合成数据的实用方法，以支持资源有限的临床文本分析。 

---
# Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings 

**Title (ZH)**: 分解“what”和“where”：使用极坐标位置嵌入 

**Authors**: Anand Gopalakrishnan, Robert Csordás, Jürgen Schmidhuber, Michael C. Mozer  

**Link**: [PDF](https://arxiv.org/pdf/2509.10534)  

**Abstract**: The attention mechanism in a Transformer architecture matches key to query based on both content -- the what -- and position in a sequence -- the where. We present an analysis indicating that what and where are entangled in the popular RoPE rotary position embedding. This entanglement can impair performance particularly when decisions require independent matches on these two factors. We propose an improvement to RoPE, which we call Polar Coordinate Position Embeddings or PoPE, that eliminates the what-where confound. PoPE is far superior on a diagnostic task requiring indexing solely by position or by content. On autoregressive sequence modeling in music, genomic, and natural language domains, Transformers using PoPE as the positional encoding scheme outperform baselines using RoPE with respect to evaluation loss (perplexity) and downstream task performance. On language modeling, these gains persist across model scale, from 124M to 774M parameters. Crucially, PoPE shows strong zero-shot length extrapolation capabilities, whereas RoPE's performance degrades significantly on longer sequences at test time without fine tuning or the use of position-interpolation methods. 

**Abstract (ZH)**: 基于极坐标位置嵌入的注意力机制在变压器架构中同时匹配查询的内容和位置。我们提出了一种分析，表明广泛使用的RoPE旋转位置嵌入中内容和位置是交织的。这种交织在需要独立匹配这两方面因素的决策中可能会影响性能。我们提出了一种改进RoPE的方法，称为极坐标位置嵌入或PoPE，它消除了内容与位置之间的混淆。PoPE在依赖位置或内容进行索引的诊断任务中表现出色。在音乐、基因组和自然语言领域的自回归序列建模中，使用PoPE作为位置编码方案的变压器在评估损失（困惑度）和下游任务性能上优于使用RoPE的基线模型。在语言建模中，这些优势在从124M到774M参数的模型规模上持续存在。最关键的是，PoPE展示了强大的零样本长度外推能力，而RoPE在其性能在对长序列进行测试时显著下降，除非进行微调或使用位置插值方法。 

---
# The Anti-Ouroboros Effect: Emergent Resilience in Large Language Models from Recursive Selective Feedback 

**Title (ZH)**: 反ouroboros效应：大型语言模型中的递归选择性反馈导致 emergent 弹性 

**Authors**: Sai Teja Reddy Adapala  

**Link**: [PDF](https://arxiv.org/pdf/2509.10509)  

**Abstract**: The stability of recursively trained large language models (LLMs) is a foundational problem for AI safety. Prevailing theory predicts model collapse, a progressive degradation when models are trained on their own output. We challenge this narrative by introducing a selective feedback mechanism. Contrary to expectation, instead of merely slowing decay, our experiments provide strong evidence that this pressure reverses it, inducing a statistically significant performance improvement in a Gemma 2B model on a complex summarization task. We name this phenomenon the Anti-Ouroboros Effect. We contrast this with a foundational experiment using a simple classifier, where the theoretical degenerative loop was validated, highlighting the unique dynamics of high-dimensional models. Our findings establish that systemic resilience can be an emergent property of LLMs under simple selection pressure, suggesting a powerful and scalable principle for developing safer and more robust AI systems. Across five generations, a quality-filtered condition improved by 6.6% in ROUGE-L F1 score, whereas an unfiltered control degraded by 3.5% and a random-filter control degraded by 4.2% 

**Abstract (ZH)**: 递归训练的大语言模型的稳定性是AI安全的基础问题。我们通过引入选择性反馈机制挑战了现有理论，实验表明这种压力不仅减缓了衰减，还显著逆转了衰减，提升了Gemma 2B模型在复杂总结任务上的性能。我们称这一现象为反奥罗boros效应。我们将这一现象与使用简单分类器的基础实验进行了对比，在后者的实验中验证了理论上的退化循环，突显了高维模型的独特动力学。我们的研究结果表明，在简单选择压力下，系统韧性可以是大语言模型的一个 Emergent 属性，这为开发更安全、更稳健的AI系统提供了一个强大且可扩展的原则。在五代训练中，经过质量过滤的条件ROUGE-L F1分数提高了6.6%，未过滤的对照组下降了3.5%，随机过滤的对照组下降了4.2%。 

---
# Learning Decomposed Contextual Token Representations from Pretrained and Collaborative Signals for Generative Recommendation 

**Title (ZH)**: 从预训练和协作信号中学习分解的上下文词表示以进行生成性推荐 

**Authors**: Yifan Liu, Yaokun Liu, Zelin Li, Zhenrui Yue, Gyuseok Lee, Ruichen Yao, Yang Zhang, Dong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.10468)  

**Abstract**: Recent advances in generative recommenders adopt a two-stage paradigm: items are first tokenized into semantic IDs using a pretrained tokenizer, and then large language models (LLMs) are trained to generate the next item via sequence-to-sequence modeling. However, these two stages are optimized for different objectives: semantic reconstruction during tokenizer pretraining versus user interaction modeling during recommender training. This objective misalignment leads to two key limitations: (i) suboptimal static tokenization, where fixed token assignments fail to reflect diverse usage contexts; and (ii) discarded pretrained semantics, where pretrained knowledge - typically from language model embeddings - is overwritten during recommender training on user interactions. To address these limitations, we propose to learn DEcomposed COntextual Token Representations (DECOR), a unified framework that preserves pretrained semantics while enhancing the adaptability of token embeddings. DECOR introduces contextualized token composition to refine token embeddings based on user interaction context, and decomposed embedding fusion that integrates pretrained codebook embeddings with newly learned collaborative embeddings. Experiments on three real-world datasets demonstrate that DECOR consistently outperforms state-of-the-art baselines in recommendation performance. Our code will be made available upon publication. 

**Abstract (ZH)**: 近期生成型推荐系统的进展采用了两阶段范式：首先使用预训练的分词器将物品 token 化为语义 ID，然后通过序列到序列建模训练大规模语言模型生成下一个物品。然而，这两阶段优化的目标不同：分词器预训练期间的语义重建与推荐器训练期间的用户交互建模。这种目标不一致性导致了两个关键限制：(i) 不理想的静态 token 化，其中固定的 token 分配无法反映不同的使用上下文；和(ii) 预训练语义的丢失，在用户交互驱动的推荐器训练中，预训练知识（通常是语言模型嵌入）被覆盖。为了解决这些限制，我们提出了一种统一框架 DEcomposed COntextual Token Representations (DECOR)，该框架保留了预训练语义的同时增强了 token 嵌入的适应性。DECOR 引入了基于用户交互上下文的 token 组合模块，以细化 token 嵌入，并结合了预训练编码本体嵌入与新学习的协作嵌入的分解嵌入融合。在三个真实世界数据集上的实验表明，DECOR 在推荐性能上始终优于最先进的基线方法。我们的代码将在发表后开源。 

---
# DSRAG: A Domain-Specific Retrieval Framework Based on Document-derived Multimodal Knowledge Graph 

**Title (ZH)**: DSRAG：基于文档衍生多模态知识图谱的领域定制检索框架 

**Authors**: Mengzheng Yang, Yanfei Ren, David Osei Opoku, Ruochang Li, Peng Ren, Chunxiao Xing  

**Link**: [PDF](https://arxiv.org/pdf/2509.10467)  

**Abstract**: Current general-purpose large language models (LLMs) commonly exhibit knowledge hallucination and insufficient domain-specific adaptability in domain-specific tasks, limiting their effectiveness in specialized question answering scenarios. Retrieval-augmented generation (RAG) effectively tackles these challenges by integrating external knowledge to enhance accuracy and relevance. However, traditional RAG still faces limitations in domain knowledge accuracy and context this http URL enhance domain-specific question answering performance, this work focuses on a graph-based RAG framework, emphasizing the critical role of knowledge graph quality during the generation process. We propose DSRAG (Domain-Specific RAG), a multimodal knowledge graph-driven retrieval-augmented generation framework designed for domain-specific applications. Our approach leverages domain-specific documents as the primary knowledge source, integrating heterogeneous information such as text, images, and tables to construct a multimodal knowledge graph covering both conceptual and instance layers. Building on this foundation, we introduce semantic pruning and structured subgraph retrieval mechanisms, combining knowledge graph context and vector retrieval results to guide the language model towards producing more reliable responses. Evaluations using the Langfuse multidimensional scoring mechanism show that our method excels in domain-specific question answering, validating the efficacy of integrating multimodal knowledge graphs with retrieval-augmented generation. 

**Abstract (ZH)**: 当前通用的大规模语言模型（LLMs）在特定领域任务中常表现出知识幻觉和领域特定适应性不足的问题，限制了其在专门化问答场景中的效果。检索增强生成（RAG）通过整合外部知识有效解决了这些挑战，增强了准确性和相关性。然而，传统的RAG仍然在领域知识准确性和上下文整合方面存在局限性。为提升领域特定问答性能，本项工作专注于图结构化的RAG框架，并强调生成过程中知识图质量的关键作用。我们提出了DSRAG（领域特定RAG），这是一种以多模态知识图驱动的检索增强生成框架，旨在服务于特定领域应用。我们的方法主要利用领域特定文档作为知识源，整合文本、图像和表格等多种异构信息，构建覆盖概念层和实例层的多模态知识图。基于此基础，我们引入了语义剪枝和结构化子图检索机制，结合知识图语境和向量检索结果来指导语言模型产生更可靠的回复。通过Langfuse多维评分机制的评估显示，我们的方法在领域特定问答任务中表现出色，验证了将多模态知识图与检索增强生成相结合的有效性。 

---
# Speaking at the Right Level: Literacy-Controlled Counterspeech Generation with RAG-RL 

**Title (ZH)**: 在合适的层级发言：基于RAG-RL的 literacy 控制式反驳生成 

**Authors**: Xiaoying Song, Anirban Saha Anik, Dibakar Barua, Pengcheng Luo, Junhua Ding, Lingzi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.01058)  

**Abstract**: Health misinformation spreading online poses a significant threat to public health. Researchers have explored methods for automatically generating counterspeech to health misinformation as a mitigation strategy. Existing approaches often produce uniform responses, ignoring that the health literacy level of the audience could affect the accessibility and effectiveness of counterspeech. We propose a Controlled-Literacy framework using retrieval-augmented generation (RAG) with reinforcement learning (RL) to generate tailored counterspeech adapted to different health literacy levels. In particular, we retrieve knowledge aligned with specific health literacy levels, enabling accessible and factual information to support generation. We design a reward function incorporating subjective user preferences and objective readability-based rewards to optimize counterspeech to the target health literacy level. Experiment results show that Controlled-Literacy outperforms baselines by generating more accessible and user-preferred counterspeech. This research contributes to more equitable and impactful public health communication by improving the accessibility and comprehension of counterspeech to health misinformation 

**Abstract (ZH)**: 健康 misinformation 在线传播对公共健康构成重大威胁。研究人员探索了自动生成针对健康 misinformation 的反制言论的方法，以减轻其影响。现有方法通常生成统一的响应，忽视了受众的健康素养水平可能会对反制言论的可达性和有效性产生影响。我们提出了一种控制健康素养水平的框架，结合检索增强生成（RAG）和强化学习（RL）来生成针对不同健康素养水平的个性化反制言论。特别是，我们检索与特定健康素养水平相符的知识，使生成的信息更具可达性和可靠性。我们设计了一种奖励函数，结合主观用户偏好和基于客观可读性的奖励，以优化针对目标健康素养水平的反制言论。实验结果表明，控制健康素养水平框架在生成更为可达和用户偏好的反制言论方面优于基线方法。此研究通过提高反制言论针对健康 misinformation 的可达性和理解度，促进了更具包容性和影响力的公共健康沟通。 

---
# A Dynamic Fusion Model for Consistent Crisis Response 

**Title (ZH)**: 一种一致性的危机响应动态融合模型 

**Authors**: Xiaoying Song, Anirban Saha Anik, Eduardo Blanco, Vanessa Frias-Martinez, Lingzi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.01053)  

**Abstract**: In response to the urgent need for effective communication with crisis-affected populations, automated responses driven by language models have been proposed to assist in crisis communications. A critical yet often overlooked factor is the consistency of response style, which could affect the trust of affected individuals in responders. Despite its importance, few studies have explored methods for maintaining stylistic consistency across generated responses. To address this gap, we propose a novel metric for evaluating style consistency and introduce a fusion-based generation approach grounded in this metric. Our method employs a two-stage process: it first assesses the style of candidate responses and then optimizes and integrates them at the instance level through a fusion process. This enables the generation of high-quality responses while significantly reducing stylistic variation between instances. Experimental results across multiple datasets demonstrate that our approach consistently outperforms baselines in both response quality and stylistic uniformity. 

**Abstract (ZH)**: 针对危机受影响人群有效沟通的迫切需求，基于语言模型的自动化响应被提出以协助危机沟通。一致性的回应风格是至关重要的但常被忽视的因素，这可能影响受影响个体对响应者的信任。尽管其重要性不言而喻，但鲜有研究探索在生成的响应之间维持风格一致性的方法。为弥补这一空白，我们提出了一种新的风格一致性评估指标，并引入了一种基于该指标的融合生成方法。我们的方法采用两阶段过程：首先评估候选回应的风格，然后通过融合过程在实例级别优化和整合它们。这使得生成高质量的回应的同时显著减少了实例间的风格变异。跨多个数据集的实验结果表明，我们的方法在响应质量和风格一致性方面均显著优于基线方法。 

---
