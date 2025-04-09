# FEABench: Evaluating Language Models on Multiphysics Reasoning Ability 

**Title (ZH)**: FEABench: 评估语言模型在多物理推理能力上的表现 

**Authors**: Nayantara Mudur, Hao Cui, Subhashini Venugopalan, Paul Raccuglia, Michael P. Brenner, Peter Norgaard  

**Link**: [PDF](https://arxiv.org/pdf/2504.06260)  

**Abstract**: Building precise simulations of the real world and invoking numerical solvers to answer quantitative problems is an essential requirement in engineering and science. We present FEABench, a benchmark to evaluate the ability of large language models (LLMs) and LLM agents to simulate and solve physics, mathematics and engineering problems using finite element analysis (FEA). We introduce a comprehensive evaluation scheme to investigate the ability of LLMs to solve these problems end-to-end by reasoning over natural language problem descriptions and operating COMSOL Multiphysics$^\circledR$, an FEA software, to compute the answers. We additionally design a language model agent equipped with the ability to interact with the software through its Application Programming Interface (API), examine its outputs and use tools to improve its solutions over multiple iterations. Our best performing strategy generates executable API calls 88% of the time. LLMs that can successfully interact with and operate FEA software to solve problems such as those in our benchmark would push the frontiers of automation in engineering. Acquiring this capability would augment LLMs' reasoning skills with the precision of numerical solvers and advance the development of autonomous systems that can tackle complex problems in the real world. The code is available at this https URL 

**Abstract (ZH)**: 构建真实世界的精确仿真并调用数值求解器以解答定量问题是工程和科学领域的基本要求。我们提出了FEABench，用于评估大型语言模型（LLMs）及其代理利用有限元分析（FEA）模拟和解决物理学、数学和工程问题的能力。我们引入了一种全面的评估方案，通过推理自然语言问题描述并操作COMSOL Multiphysics®（一种FEA软件），来调查LLMs端到端解决这些问题的能力。此外，我们设计了一个具备与软件交互能力的语言模型代理，并通过评估其输出和使用工具在多轮迭代中改进其解决方案。我们的最佳策略88%的时间生成可执行的API调用。能够成功与FEA软件交互并解决基准中问题的LLMs，将推动工程自动化领域的边界。获得这一能力将增强LLMs的推理技能，与数值求解器的精确性相结合，推进能够应对现实世界复杂问题的自主系统的开发。代码可在以下链接获取：this https URL 

---
# TxGemma: Efficient and Agentic LLMs for Therapeutics 

**Title (ZH)**: TxGemma: 高效自主的治疗用语言模型 

**Authors**: Eric Wang, Samuel Schmidgall, Paul F. Jaeger, Fan Zhang, Rory Pilgrim, Yossi Matias, Joelle Barral, David Fleet, Shekoofeh Azizi  

**Link**: [PDF](https://arxiv.org/pdf/2504.06196)  

**Abstract**: Therapeutic development is a costly and high-risk endeavor that is often plagued by high failure rates. To address this, we introduce TxGemma, a suite of efficient, generalist large language models (LLMs) capable of therapeutic property prediction as well as interactive reasoning and explainability. Unlike task-specific models, TxGemma synthesizes information from diverse sources, enabling broad application across the therapeutic development pipeline. The suite includes 2B, 9B, and 27B parameter models, fine-tuned from Gemma-2 on a comprehensive dataset of small molecules, proteins, nucleic acids, diseases, and cell lines. Across 66 therapeutic development tasks, TxGemma achieved superior or comparable performance to the state-of-the-art generalist model on 64 (superior on 45), and against state-of-the-art specialist models on 50 (superior on 26). Fine-tuning TxGemma models on therapeutic downstream tasks, such as clinical trial adverse event prediction, requires less training data than fine-tuning base LLMs, making TxGemma suitable for data-limited applications. Beyond these predictive capabilities, TxGemma features conversational models that bridge the gap between general LLMs and specialized property predictors. These allow scientists to interact in natural language, provide mechanistic reasoning for predictions based on molecular structure, and engage in scientific discussions. Building on this, we further introduce Agentic-Tx, a generalist therapeutic agentic system powered by Gemini 2.5 that reasons, acts, manages diverse workflows, and acquires external domain knowledge. Agentic-Tx surpasses prior leading models on the Humanity's Last Exam benchmark (Chemistry & Biology) with 52.3% relative improvement over o3-mini (high) and 26.7% over o3-mini (high) on GPQA (Chemistry) and excels with improvements of 6.3% (ChemBench-Preference) and 2.4% (ChemBench-Mini) over o3-mini (high). 

**Abstract (ZH)**: 疗法治开发是一项成本高、风险大的过程，往往伴随着高失败率。为解决这一问题，我们引入了TxGemma，这是一个高效的、通用的大语言模型套件，能够进行治疗性质预测、交互推理和可解释性分析。与任务特定模型不同，TxGemma能够综合多种来源的信息，使其在治疗开发管道的各个阶段具有广泛的应用。该套件包括2亿、9亿和27亿参数的模型，这些模型基于Gemmar-2，在涉及小分子、蛋白质、核酸、疾病和细胞系的全面数据集上进行微调。在66个治疗开发任务中，TxGemma在64个任务上优于或达到了最先进的通用模型的最佳性能（其中45个任务表现更优），在50个任务上优于最先进的专用模型（其中26个任务表现更优）。针对诸如临床试验不良事件预测等治疗下游任务微调TxGemma模型需要比微调基础大语言模型更少的训练数据，使得TxGemma适用于数据受限的应用。除了预测能力之外，TxGemma还配备了对话模型，填补了一般大语言模型和专门属性预测器之间的差距。这些模型允许科学家自然语言交流，根据分子结构提供机制推理，并参与科学讨论。在此基础上，我们进一步引入了由Gemini 2.5驱动的通用治疗决策系统Agentic-Tx，该系统能够进行推理、行动、管理多样化的工作流，并获取外部专业知识。Agentic-Tx在人类最终考试基准（ Chemistry & Biology）上超越了之前的领先模型，相对改进了52.3%（相对于o3-mini 高版本），在GPQA（ Chemistry）上相对改进了26.7%（相对于o3-mini 高版本），在ChemBench-Preference和ChemBench-Mini上分别取得了6.3%和2.4%的改进（相对于o3-mini 高版本）。 

---
# Leanabell-Prover: Posttraining Scaling in Formal Reasoning 

**Title (ZH)**: Leanabell-Prover: 训练后权重缩放在形式推理中的应用 

**Authors**: Jingyuan Zhang, Qi Wang, Xingguang Ji, Yahui Liu, Yang Yue, Fuzheng Zhang, Di Zhang, Guorui Zhou, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2504.06122)  

**Abstract**: Recent advances in automated theorem proving (ATP) through LLMs have highlighted the potential of formal reasoning with Lean 4 codes. However, ATP has not yet be revolutionized by the recent posttraining scaling as demonstrated by Open AI O1/O3 and Deepseek R1. In this work, we investigate the entire posttraining of ATP, aiming to align it with breakthroughs in reasoning models in natural this http URL begin, we continual train current ATP models with a hybrid dataset, which consists of numerous statement-proof pairs, and additional data aimed at incorporating cognitive behaviors that emulate human reasoning and hypothesis refinement. Next, we explore reinforcement learning with the use of outcome reward returned by Lean 4 compiler. Through our designed continual training and reinforcement learning processes, we have successfully improved existing formal provers, including both DeepSeek-Prover-v1.5 and Goedel-Prover, achieving state-of-the-art performance in the field of whole-proof generation. For example, we achieve a 59.8% pass rate (pass@32) on MiniF2F. This is an on-going project and we will progressively update our findings, release our data and training details. 

**Abstract (ZH)**: Recent advances in automated theorem proving through LLMs have highlighted the potential of formal reasoning with Lean 4 codes. However, ATP has not yet been revolutionized by recent posttraining scaling as demonstrated by Open AI O1/O3 and Deepseek R1. In this work, we investigate the entire posttraining of ATP, aiming to align it with breakthroughs in reasoning models in natural language processing. To begin, we continually train current ATP models with a hybrid dataset, which consists of numerous statement-proof pairs, and additional data aimed at incorporating cognitive behaviors that emulate human reasoning and hypothesis refinement. Next, we explore reinforcement learning with the use of outcome reward returned by Lean 4 compiler. Through our designed continual training and reinforcement learning processes, we have successfully improved existing formal provers, including both DeepSeek-Prover-v1.5 and Goedel-Prover, achieving state-of-the-art performance in the field of whole-proof generation. For example, we achieve a 59.8% pass rate (pass@32) on MiniF2F. This is an on-going project and we will progressively update our findings, release our data and training details. 

---
# Agent Guide: A Simple Agent Behavioral Watermarking Framework 

**Title (ZH)**: 智能体引导：一种简单的智能体行为水印框架 

**Authors**: Kaibo Huang, Zhongliang Yang, Linna Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.05871)  

**Abstract**: The increasing deployment of intelligent agents in digital ecosystems, such as social media platforms, has raised significant concerns about traceability and accountability, particularly in cybersecurity and digital content protection. Traditional large language model (LLM) watermarking techniques, which rely on token-level manipulations, are ill-suited for agents due to the challenges of behavior tokenization and information loss during behavior-to-action translation. To address these issues, we propose Agent Guide, a novel behavioral watermarking framework that embeds watermarks by guiding the agent's high-level decisions (behavior) through probability biases, while preserving the naturalness of specific executions (action). Our approach decouples agent behavior into two levels, behavior (e.g., choosing to bookmark) and action (e.g., bookmarking with specific tags), and applies watermark-guided biases to the behavior probability distribution. We employ a z-statistic-based statistical analysis to detect the watermark, ensuring reliable extraction over multiple rounds. Experiments in a social media scenario with diverse agent profiles demonstrate that Agent Guide achieves effective watermark detection with a low false positive rate. Our framework provides a practical and robust solution for agent watermarking, with applications in identifying malicious agents and protecting proprietary agent systems. 

**Abstract (ZH)**: 智能代理在数字生态系统中的不断增加部署引发了对跟踪性和问责性的重大关注，特别是在网络安全和数字内容保护方面。传统的基于令牌级操纵的大型语言模型（LLM）水印技术不适合智能代理，因为行为到操作转换过程中存在行为标记化和信息丢失的挑战。为了解决这些问题，我们提出了一种新的行为水印框架——Agent Guide，通过概率偏差指导智能代理的高层次决策（行为），同时保持特定执行（操作）的自然性。我们的方法将智能代理行为分为两个层次，行为（例如，选择书签）和操作（例如，使用特定标签进行书签操作），并应用于行为概率分布的水印指导偏差。我们采用基于z统计量的统计分析来检测水印，确保在多轮中可靠地提取。在具有多样智能代理配置的社交媒体场景中进行的实验表明，Agent Guide能够以较低的假阳性率有效地检测水印。该框架为智能代理水印提供了一种实用和 robust 的解决方案，应用于识别恶意代理和保护专有代理系统。 

---
# Are Generative AI Agents Effective Personalized Financial Advisors? 

**Title (ZH)**: 生成式AI代理是否有效的个性化金融顾问？ 

**Authors**: Takehiro Takayanagi, Kiyoshi Izumi, Javier Sanz-Cruzado, Richard McCreadie, Iadh Ounis  

**Link**: [PDF](https://arxiv.org/pdf/2504.05862)  

**Abstract**: Large language model-based agents are becoming increasingly popular as a low-cost mechanism to provide personalized, conversational advice, and have demonstrated impressive capabilities in relatively simple scenarios, such as movie recommendations. But how do these agents perform in complex high-stakes domains, where domain expertise is essential and mistakes carry substantial risk? This paper investigates the effectiveness of LLM-advisors in the finance domain, focusing on three distinct challenges: (1) eliciting user preferences when users themselves may be unsure of their needs, (2) providing personalized guidance for diverse investment preferences, and (3) leveraging advisor personality to build relationships and foster trust. Via a lab-based user study with 64 participants, we show that LLM-advisors often match human advisor performance when eliciting preferences, although they can struggle to resolve conflicting user needs. When providing personalized advice, the LLM was able to positively influence user behavior, but demonstrated clear failure modes. Our results show that accurate preference elicitation is key, otherwise, the LLM-advisor has little impact, or can even direct the investor toward unsuitable assets. More worryingly, users appear insensitive to the quality of advice being given, or worse these can have an inverse relationship. Indeed, users reported a preference for and increased satisfaction as well as emotional trust with LLMs adopting an extroverted persona, even though those agents provided worse advice. 

**Abstract (ZH)**: 基于大型语言模型的顾问在金融领域的有效性：应对三大挑战 

---
# From Superficial to Deep: Integrating External Knowledge for Follow-up Question Generation Using Knowledge Graph and LLM 

**Title (ZH)**: 从表层到深度：利用知识图谱和大语言模型进行跟进问题生成的外部知识集成方法 

**Authors**: Jianyu Liu, Yi Huang, Sheng Bi, Junlan Feng, Guilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2504.05801)  

**Abstract**: In a conversational system, dynamically generating follow-up questions based on context can help users explore information and provide a better user experience. Humans are usually able to ask questions that involve some general life knowledge and demonstrate higher order cognitive skills. However, the questions generated by existing methods are often limited to shallow contextual questions that are uninspiring and have a large gap to the human level. In this paper, we propose a three-stage external knowledge-enhanced follow-up question generation method, which generates questions by identifying contextual topics, constructing a knowledge graph (KG) online, and finally combining these with a large language model to generate the final question. The model generates information-rich and exploratory follow-up questions by introducing external common sense knowledge and performing a knowledge fusion operation. Experiments show that compared to baseline models, our method generates questions that are more informative and closer to human questioning levels while maintaining contextual relevance. 

**Abstract (ZH)**: 基于上下文的外部知识增强型跟随问题生成方法：多层次生成启发用户探索信息的问题 

---
# Automated Archival Descriptions with Federated Intelligence of LLMs 

**Title (ZH)**: 联邦大语言模型智能自动化档案描述 

**Authors**: Jinghua Groppe, Andreas Marquet, Annabel Walz, Sven Groppe  

**Link**: [PDF](https://arxiv.org/pdf/2504.05711)  

**Abstract**: Enforcing archival standards requires specialized expertise, and manually creating metadata descriptions for archival materials is a tedious and error-prone task. This work aims at exploring the potential of agentic AI and large language models (LLMs) in addressing the challenges of implementing a standardized archival description process. To this end, we introduce an agentic AI-driven system for automated generation of high-quality metadata descriptions of archival materials. We develop a federated optimization approach that unites the intelligence of multiple LLMs to construct optimal archival metadata. We also suggest methods to overcome the challenges associated with using LLMs for consistent metadata generation. To evaluate the feasibility and effectiveness of our techniques, we conducted extensive experiments using a real-world dataset of archival materials, which covers a variety of document types and data formats. The evaluation results demonstrate the feasibility of our techniques and highlight the superior performance of the federated optimization approach compared to single-model solutions in metadata quality and reliability. 

**Abstract (ZH)**: 实施存档标准需要专门的知识和技能，手动创建存档材料的元数据描述是一个繁琐且容易出错的任务。本研究旨在探讨自主人工智能（agentic AI）和大型语言模型（LLMs）在解决实施标准化存档描述流程挑战方面的潜力。为此，我们提出了一种基于自主人工智能的自动化生成高质量存档材料元数据描述的系统。我们开发了一种联邦优化方法，将多个LLM的智能结合起来以构建最优存档元数据。我们还提出了克服使用LLMs进行一致的元数据生成所面临挑战的方法。为了评估我们技术的可行性和有效性，我们使用包含各种文件类型和数据格式的真实世界存档材料数据集进行了广泛的实验。评估结果表明了我们技术的可行性，并强调了联邦优化方法在元数据质量和可靠性方面的优越性能，优于单一模型解决方案。 

---
# SciSciGPT: Advancing Human-AI Collaboration in the Science of Science 

**Title (ZH)**: SciSciGPT：推动科学学中的人类-人工智能协作 

**Authors**: Erzhuo Shao, Yifang Wang, Yifan Qian, Zhenyu Pan, Han Liu, Dashun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05559)  

**Abstract**: The increasing availability of large-scale datasets has fueled rapid progress across many scientific fields, creating unprecedented opportunities for research and discovery while posing significant analytical challenges. Recent advances in large language models (LLMs) and AI agents have opened new possibilities for human-AI collaboration, offering powerful tools to navigate this complex research landscape. In this paper, we introduce SciSciGPT, an open-source, prototype AI collaborator that uses the science of science as a testbed to explore the potential of LLM-powered research tools. SciSciGPT automates complex workflows, supports diverse analytical approaches, accelerates research prototyping and iteration, and facilitates reproducibility. Through case studies, we demonstrate its ability to streamline a wide range of empirical and analytical research tasks while highlighting its broader potential to advance research. We further propose an LLM Agent capability maturity model for human-AI collaboration, envisioning a roadmap to further improve and expand upon frameworks like SciSciGPT. As AI capabilities continue to evolve, frameworks like SciSciGPT may play increasingly pivotal roles in scientific research and discovery, unlocking further opportunities. At the same time, these new advances also raise critical challenges, from ensuring transparency and ethical use to balancing human and AI contributions. Addressing these issues may shape the future of scientific inquiry and inform how we train the next generation of scientists to thrive in an increasingly AI-integrated research ecosystem. 

**Abstract (ZH)**: 大规模数据集的不断增加促进了多个科学领域的迅速进步，创造了前所未有的研究与发现机会，同时也带来了重大的分析挑战。大型语言模型（LLMs）和AI代理的最新进展为人类与AI的合作开辟了新的可能性，提供了强大的工具以应对这一复杂的研究环境。本文介绍了SciSciGPT，这是一种开源原型AI合作者，利用科学的科学研究作为试验场，探索LLM驱动的研究工具的潜力。SciSciGPT自动化复杂的工作流，支持多样化的分析方法，加速研究原型设计和迭代，并促进可重复性。通过案例研究，我们展示了它在一系列实证和分析研究任务中简化流程的能力，并阐明了其更广泛的潜力以推动研究发展。我们进一步提出了AI代理能力成熟模型，展望了改进和扩展如SciSciGPT等框架的道路。随着AI能力的不断演进，框架如SciSciGPT可能在科学研究与发现中发挥越来越关键的作用，解锁新的机会。与此同时，这些新进展也引发了重要的挑战，从确保透明性和伦理使用到平衡人类和AI的贡献。解决这些问题可能会影响科学研究的未来，并指导我们如何培养下一代科学家在日益集成AI的研究生态系统中茁壮成长。 

---
# Prism: Dynamic and Flexible Benchmarking of LLMs Code Generation with Monte Carlo Tree Search 

**Title (ZH)**: Prism: 基于蒙特卡洛树搜索的LLM代码生成动态灵活基准测试 

**Authors**: Vahid Majdinasab, Amin Nikanjam, Foutse Khomh  

**Link**: [PDF](https://arxiv.org/pdf/2504.05500)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has outpaced traditional evaluation methods. Static benchmarks fail to capture the depth and breadth of LLM capabilities and eventually become obsolete, while most dynamic approaches either rely too heavily on LLM-based evaluation or remain constrained by predefined test sets. We introduce Prism, a flexible, dynamic benchmarking framework designed for comprehensive LLM assessment. Prism builds on three key components: (1) a tree-based state representation that models evaluation as a Markov Decision Process, (2) a Monte Carlo Tree Search algorithm adapted to uncover challenging evaluation scenarios, and (3) a multi-agent evaluation pipeline that enables simultaneous assessment of diverse capabilities. To ensure robust evaluation, Prism integrates structural measurements of tree exploration patterns with performance metrics across difficulty levels, providing detailed diagnostics of error patterns, test coverage, and solution approaches. Through extensive experiments on five state-of-the-art LLMs, we analyze how model architecture and scale influence code generation performance across varying task difficulties. Our results demonstrate Prism's effectiveness as a dynamic benchmark that evolves with model advancements while offering deeper insights into their limitations. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅速发展超出了传统评价方法的速度。静态基准无法捕捉LLM的能力深度和广度，最终变得过时，而大多数动态方法要么过于依赖LLM评价，要么受限于预定义的测试集。我们引入了Prism，一个灵活的动态基准框架，用于全面评估LLM。Prism基于三个关键组件构建：（1）一种基于树的状态表示，将评价建模为马尔可夫决策过程；（2）一种适应性蒙特卡洛树搜索算法，用于揭示具有挑战性的评价场景；（3）一种多代理评价流水线，可同时评估多种能力。为确保评估的稳健性，Prism将树探索模式的结构测量与不同难度级别的性能指标集成，提供详细的错误模式、测试覆盖和解题方法诊断。通过在五种先进LLM上的广泛实验，我们分析了模型架构和规模对不同任务难度下的代码生成性能影响。我们的结果表明，Prism是一个随着模型进步而不断进化、并提供深入见解的动态基准。 

---
# EduPlanner: LLM-Based Multi-Agent Systems for Customized and Intelligent Instructional Design 

**Title (ZH)**: EduPlanner: 基于大语言模型的多代理系统，实现个性化和智能的教学设计 

**Authors**: Xueqiao Zhang, Chao Zhang, Jianwen Sun, Jun Xiao, Yi Yang, Yawei Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.05370)  

**Abstract**: Large Language Models (LLMs) have significantly advanced smart education in the Artificial General Intelligence (AGI) era. A promising application lies in the automatic generalization of instructional design for curriculum and learning activities, focusing on two key aspects: (1) Customized Generation: generating niche-targeted teaching content based on students' varying learning abilities and states, and (2) Intelligent Optimization: iteratively optimizing content based on feedback from learning effectiveness or test scores. Currently, a single large LLM cannot effectively manage the entire process, posing a challenge for designing intelligent teaching plans. To address these issues, we developed EduPlanner, an LLM-based multi-agent system comprising an evaluator agent, an optimizer agent, and a question analyst, working in adversarial collaboration to generate customized and intelligent instructional design for curriculum and learning activities. Taking mathematics lessons as our example, EduPlanner employs a novel Skill-Tree structure to accurately model the background mathematics knowledge of student groups, personalizing instructional design for curriculum and learning activities according to students' knowledge levels and learning abilities. Additionally, we introduce the CIDDP, an LLM-based five-dimensional evaluation module encompassing clarity, Integrity, Depth, Practicality, and Pertinence, to comprehensively assess mathematics lesson plan quality and bootstrap intelligent optimization. Experiments conducted on the GSM8K and Algebra datasets demonstrate that EduPlanner excels in evaluating and optimizing instructional design for curriculum and learning activities. Ablation studies further validate the significance and effectiveness of each component within the framework. Our code is publicly available at this https URL 

**Abstract (ZH)**: 大规模语言模型（LLMs）在人工智能通用时代（AGI）显著推动了智能教育的发展。一种有前途的应用在于基于自动化的教学设计的一般化，主要集中在两个关键方面：（1）个性化生成：根据学生不同的学习能力和状态生成针对特定需求的教学内容，（2）智能优化：根据学习效果或测试分数的反馈迭代优化内容。当前，单个大规模语言模型无法有效管理整个过程，为设计智能教学计划带来了挑战。为解决这些问题，我们开发了EduPlanner，一个基于大规模语言模型的多智能体系统，包括评估代理、优化代理和题库分析师，它们以对抗性合作的方式工作，为教学计划和学习活动生成个性化和智能化的教学设计。以数学课程为例，EduPlanner采用了一种新颖的技能树结构，准确建模学生群体的背景数学知识，并根据学生的知识水平和学习能力个性化教学设计。此外，我们引入了CIDDP，一种基于大规模语言模型的五维评估模块，涵盖清晰性、完整性、深度、实践性和相关性，全面评估数学课程计划的质量并启动智能优化。在GSM8K和代数数据集上的实验表明，EduPlanner在评估和优化教学设计方面表现出色。消融研究表明，框架内的每个组件的重要性及其有效性。我们的代码已公开在这个网址。 

---
# GOLLuM: Gaussian Process Optimized LLMs -- Reframing LLM Finetuning through Bayesian Optimization 

**Title (ZH)**: GOLLuM：高斯过程优化的大语言模型——通过贝叶斯优化重新构想大语言模型微调 

**Authors**: Bojana Ranković, Philippe Schwaller  

**Link**: [PDF](https://arxiv.org/pdf/2504.06265)  

**Abstract**: Large Language Models (LLMs) can encode complex relationships in their latent spaces, yet harnessing them for optimization under uncertainty remains challenging. We address this gap with a novel architecture that reframes LLM finetuning as Gaussian process (GP) marginal likelihood optimization via deep kernel methods. We introduce LLM-based deep kernels, jointly optimized with GPs to preserve the benefits of both - LLMs to provide a rich and flexible input space for Bayesian optimization and - GPs to model this space with predictive uncertainty for more efficient sampling. Applied to Buchwald-Hartwig reaction optimization, our method nearly doubles the discovery rate of high-performing reactions compared to static LLM embeddings (from 24% to 43% coverage of the top 5% reactions in just 50 optimization iterations). We also observe a 14% improvement over domain-specific representations without requiring specialized features. Extensive empirical evaluation across 19 benchmarks - ranging from general chemistry to reaction and molecular property optimization - demonstrates our method's robustness, generality, and consistent improvements across: (1) tasks, (2) LLM architectures (encoder, decoder, encoder-decoder), (3) pretraining domains (chemistry-related or general-purpose) and (4) hyperparameter settings (tuned once on a single dataset). Finally, we explain these improvements: joint LLM-GP optimization through marginal likelihood implicitly performs contrastive learning, aligning representations to produce (1) better-structured embedding spaces, (2) improved uncertainty calibration, and (3) more efficient sampling - without requiring any external loss. This work provides both practical advances in sample-efficient optimization and insights into what makes effective Bayesian optimization. 

**Abstract (ZH)**: 大规模语言模型（LLMs）能够在其潜在空间中编码复杂的关系，但将其用于不确定性的优化仍然具有挑战性。我们通过深度核方法将LLM微调重新框架为高斯过程（GP）边际似然优化，提出了一种新颖的架构。我们引入了基于LLM的深度核，并与GP联合优化，以保留两者的优点——LLM提供丰富的灵活输入空间以供贝叶斯优化使用，GP则使用预测不确定性来建模该空间以实现更高效的采样。应用于Buchwald-Hartwig反应优化，我们的方法在50次优化迭代中将高表现反应的发现率几乎翻了一番（从静态LLM嵌入中的24%提高到43%的前5%反应覆盖率）。我们还观察到在不需要专门特征的情况下，对领域特定表示有14%的改进。跨19个基准的广泛实证评估涵盖了从通用化学到反应和分子性质优化，证实了该方法的稳健性、通用性和在以下方面的持续改进：（1）任务，（2）LLM架构（编码器、解码器、编码器-解码器），（3）预训练领域（化学相关或通用用途）和（4）超参数设置（在单一数据集中调整一次）。最后，我们解释了这些改进：联合LLM-GP优化通过边际似然隐式执行对比学习，使表示对齐以产生（1）更好的结构化嵌入空间，（2）更好的不确定性校准，以及（3）更高效的采样——而无需任何外部损失。该工作不仅提供了样本高效优化的实际进展，还揭示了有效贝叶斯优化的内在机制。 

---
# From 128K to 4M: Efficient Training of Ultra-Long Context Large Language Models 

**Title (ZH)**: 从128K到4M：高效训练超长上下文大语言模型 

**Authors**: Chejian Xu, Wei Ping, Peng Xu, Zihan Liu, Boxin Wang, Mohammad Shoeybi, Bo Li, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2504.06214)  

**Abstract**: Long-context capabilities are essential for a wide range of applications, including document and video understanding, in-context learning, and inference-time scaling, all of which require models to process and reason over long sequences of text and multimodal data. In this work, we introduce a efficient training recipe for building ultra-long context LLMs from aligned instruct model, pushing the boundaries of context lengths from 128K to 1M, 2M, and 4M tokens. Our approach leverages efficient continued pretraining strategies to extend the context window and employs effective instruction tuning to maintain the instruction-following and reasoning abilities. Our UltraLong-8B, built on Llama3.1-Instruct with our recipe, achieves state-of-the-art performance across a diverse set of long-context benchmarks. Importantly, models trained with our approach maintain competitive performance on standard benchmarks, demonstrating balanced improvements for both long and short context tasks. We further provide an in-depth analysis of key design choices, highlighting the impacts of scaling strategies and data composition. Our findings establish a robust framework for efficiently scaling context lengths while preserving general model capabilities. We release all model weights at: this https URL. 

**Abstract (ZH)**: 长上下文能力对于文档和视频理解、上下文学习以及推理时的扩展等多种应用至关重要，这些应用要求模型能够处理和推理长文本和多模态数据。在本工作中，我们提出了一种高效的训练方法，用于构建超长上下文语言模型，将上下文长度从128K扩展到1M、2M和4M tokens。我们的方法利用高效的持续预训练策略扩展上下文窗口，并采用有效的指令微调保持指令遵循和推理能力。基于我们的方法构建的UltraLong-8B在多种长上下文基准测试中取得了最佳性能。重要的是，使用我们方法训练的模型在标准基准测试中保持了竞争力，证明了对长上下文和短上下文任务的均衡改进。我们还对关键设计选择进行了深入分析，强调了扩展策略和数据组成的影响。我们的发现建立了一个在高效扩展上下文长度的同时保留通用模型能力的稳健框架。所有模型权重已发布于：this https URL。 

---
# Navigating the Rabbit Hole: Emergent Biases in LLM-Generated Attack Narratives Targeting Mental Health Groups 

**Title (ZH)**: 穿越兔子洞：大语言模型生成的针对心理健康群体的攻击性叙事中的涌现偏差 

**Authors**: Rijul Magu, Arka Dutta, Sean Kim, Ashiqur R. KhudaBukhsh, Munmun De Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2504.06160)  

**Abstract**: Large Language Models (LLMs) have been shown to demonstrate imbalanced biases against certain groups. However, the study of unprovoked targeted attacks by LLMs towards at-risk populations remains underexplored. Our paper presents three novel contributions: (1) the explicit evaluation of LLM-generated attacks on highly vulnerable mental health groups; (2) a network-based framework to study the propagation of relative biases; and (3) an assessment of the relative degree of stigmatization that emerges from these attacks. Our analysis of a recently released large-scale bias audit dataset reveals that mental health entities occupy central positions within attack narrative networks, as revealed by a significantly higher mean centrality of closeness (p-value = 4.06e-10) and dense clustering (Gini coefficient = 0.7). Drawing from sociological foundations of stigmatization theory, our stigmatization analysis indicates increased labeling components for mental health disorder-related targets relative to initial targets in generation chains. Taken together, these insights shed light on the structural predilections of large language models to heighten harmful discourse and highlight the need for suitable approaches for mitigation. 

**Abstract (ZH)**: 大型语言模型（LLMs）已被证明对某些群体表现出不平衡的偏见。然而，LLMs对脆弱群体进行无缘无故的针对性攻击的研究仍然欠探索。本文提出三大新颖贡献：（1）明确评估LLM生成的针对精神健康高度脆弱群体的攻击；（2）基于网络的框架以研究相对偏见的传播；（3）评估这些攻击中精神健康领域所引起的相对污名化程度。我们分析一个新发布的大型偏见审计数据集表明，精神健康实体在攻击叙事网络中占据中心位置，显示出较高的平均接近性中心度（p值=4.06e-10）和密集聚类（基尼系数=0.7）。基于污名化理论的社会学基础，我们的污名化分析显示，精神健康障碍相关目标在生成链中的标签成分增加幅度大于初始目标。综合这些见解，它们揭示了大型语言模型在加剧有害话语方面的结构性倾向，并突显了需要采取适当方法进行缓解的必要性。 

---
# ARLO: A Tailorable Approach for Transforming Natural Language Software Requirements into Architecture using LLMs 

**Title (ZH)**: ARLO：一种使用大语言模型将自然语言软件需求转换为架构的可配置方法 

**Authors**: Tooraj Helmi  

**Link**: [PDF](https://arxiv.org/pdf/2504.06143)  

**Abstract**: Software requirements expressed in natural language (NL) frequently suffer from verbosity, ambiguity, and inconsistency. This creates a range of challenges, including selecting an appropriate architecture for a system and assessing different architectural alternatives. Relying on human expertise to accomplish the task of mapping NL requirements to architecture is time-consuming and error-prone. This paper proposes ARLO, an approach that automates this task by leveraging (1) a set of NL requirements for a system, (2) an existing standard that specifies architecturally relevant software quality attributes, and (3) a readily available Large Language Model (LLM). Specifically, ARLO determines the subset of NL requirements for a given system that is architecturally relevant and maps that subset to a tailorable matrix of architectural choices. ARLO applies integer linear programming on the architectural-choice matrix to determine the optimal architecture for the current requirements. We demonstrate ARLO's efficacy using a set of real-world examples. We highlight ARLO's ability (1) to trace the selected architectural choices to the requirements and (2) to isolate NL requirements that exert a particular influence on a system's architecture. This allows the identification, comparative assessment, and exploration of alternative architectural choices based on the requirements and constraints expressed therein. 

**Abstract (ZH)**: 自然语言表达的软件需求经常存在冗长、模糊和不一致的问题，这给选择合适的系统架构及评估不同的架构选项带来了挑战。依赖人类专业知识将自然语言需求映射到架构的过程耗时且易出错。本文提出ARLO方法，通过利用（1）系统的自然语言需求集、（2）一个现有的规范，该规范规定了与架构相关的软件质量属性，以及（3）一个现成的大规模语言模型（LLM），自动化这一任务。ARLO确定给定系统中与架构相关的自然语言需求子集，并将其映射到可定制的架构选择矩阵。ARLO通过对架构选择矩阵应用整数线性规划来确定当前需求的最优架构。我们使用一组实际案例展示了ARLO的有效性，并突出了ARLO的能力，即（1）追踪所选架构选择与需求的关系，以及（2）隔离对系统架构具有特定影响的自然语言需求，从而根据其中表达的需求和约束条件进行架构选择的标识、比较评估和探索。 

---
# QGen Studio: An Adaptive Question-Answer Generation, Training and Evaluation Platform 

**Title (ZH)**: QGen Studio: 一种自适应问题-答案生成、训练与评估平台 

**Authors**: Movina Moses, Mohab Elkaref, James Barry, Shinnosuke Tanaka, Vishnudev Kuruvanthodi, Nathan Herr, Campbell D Watson, Geeth De Mel  

**Link**: [PDF](https://arxiv.org/pdf/2504.06136)  

**Abstract**: We present QGen Studio: an adaptive question-answer generation, training, and evaluation platform. QGen Studio enables users to leverage large language models (LLMs) to create custom question-answer datasets and fine-tune models on this synthetic data. It features a dataset viewer and model explorer to streamline this process. The dataset viewer provides key metrics and visualizes the context from which the QA pairs are generated, offering insights into data quality. The model explorer supports model comparison, allowing users to contrast the performance of their trained LLMs against other models, supporting performance benchmarking and refinement. QGen Studio delivers an interactive, end-to-end solution for generating QA datasets and training scalable, domain-adaptable models. The studio will be open-sourced soon, allowing users to deploy it locally. 

**Abstract (ZH)**: QGen Studio: 一种自适应的问答生成、训练和评估平台 

---
# Confidence Regularized Masked Language Modeling using Text Length 

**Title (ZH)**: 长度正则化掩蔽语言模型中的信心调节 

**Authors**: Seunghyun Ji, Soowon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.06037)  

**Abstract**: Masked language modeling, which is a task to predict a randomly masked word in the input text, is an efficient language representation learning method. Masked language modeling ignores various words which people can think of for filling in the masked position and calculates the loss with a single word. Especially when the input text is short, the entropy of the word distribution that can fill in the masked position can be high. This may cause the model to be overconfident in the single answer. To address this issue, we propose a novel confidence regularizer that controls regularizing strength dynamically by the input text length. Experiments with GLUE and SQuAD datasets showed that our method achieves better accuracy and lower expected calibration error. 

**Abstract (ZH)**: 掩码语言模型是一种用于预测输入文本中随机掩蔽词的高效语言表示学习方法。掩码语言模型忽略了人们可以用来填充掩蔽位置的各种词语，并使用单个词语计算损失。尤其是在输入文本较短时，可以填充掩蔽位置的词分布的熵可能会很高，这可能导致模型对单一答案过于自信。为解决这一问题，我们提出了一种新的置信度正则化方法，通过动态调整正则化强度来控制模型的置信度。实验结果表明，我们的方法在GLUE和SQuAD数据集上取得了更高的准确率和更低的预期校准误差。 

---
# Optuna vs Code Llama: Are LLMs a New Paradigm for Hyperparameter Tuning? 

**Title (ZH)**: Optuna vs Code Llama：大规模语言模型是否成为超参数调优的新范式？ 

**Authors**: Roman Kochnev, Arash Torabi Goodarzi, Zofia Antonina Bentyn, Dmitry Ignatov, Radu Timofte  

**Link**: [PDF](https://arxiv.org/pdf/2504.06006)  

**Abstract**: Optimal hyperparameter selection is critical for maximizing neural network performance, especially as models grow in complexity. This work investigates the viability of using large language models (LLMs) for hyperparameter optimization by employing a fine-tuned version of Code Llama. Through parameter-efficient fine-tuning using LoRA, we adapt the LLM to generate accurate and efficient hyperparameter recommendations tailored to diverse neural network architectures. Unlike traditional methods such as Optuna, which rely on exhaustive trials, the proposed approach achieves competitive or superior results in terms of Root Mean Square Error (RMSE) while significantly reducing computational overhead. Our approach highlights that LLM-based optimization not only matches state-of-the-art methods like Tree-structured Parzen Estimators but also accelerates the tuning process. This positions LLMs as a promising alternative to conventional optimization techniques, particularly for rapid experimentation. Furthermore, the ability to generate hyperparameters in a single inference step makes this method particularly well-suited for resource-constrained environments such as edge devices and mobile applications, where computational efficiency is paramount. The results confirm that LLMs, beyond their efficiency, offer substantial time savings and comparable stability, underscoring their value in advancing machine learning workflows. All generated hyperparameters are included in the LEMUR Neural Network (NN) Dataset, which is publicly available and serves as an open-source benchmark for hyperparameter optimization research. 

**Abstract (ZH)**: 使用大型语言模型进行超参数优化：Code Llama的参数高效微调技术 

---
# NativQA Framework: Enabling LLMs with Native, Local, and Everyday Knowledge 

**Title (ZH)**: NativQA框架：使大语言模型蕴含本地化、日常知识能力 

**Authors**: Firoj Alam, Md Arid Hasan, Sahinur Rahman Laskar, Mucahid Kutlu, Shammur Absar Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2504.05995)  

**Abstract**: The rapid advancement of large language models (LLMs) has raised concerns about cultural bias, fairness, and their applicability in diverse linguistic and underrepresented regional contexts. To enhance and benchmark the capabilities of LLMs, there is a need to develop large-scale resources focused on multilingual, local, and cultural contexts. In this study, we propose a framework, NativQA, that can seamlessly construct large-scale, culturally and regionally aligned QA datasets in native languages. The framework utilizes user-defined seed queries and leverages search engines to collect location-specific, everyday information. It has been evaluated across 39 locations in 24 countries and in 7 languages, ranging from extremely low-resource to high-resource languages, which resulted over 300K Question Answer (QA) pairs. The developed resources can be used for LLM benchmarking and further fine-tuning. The framework has been made publicly available for the community (this https URL). 

**Abstract (ZH)**: 大规模语言模型的迅速进步引发了对其文化偏见、公平性及其在多元语言和欠代表地区应用场景的顾虑。为提升并评估语言模型的能力，有必要开发专注于多语言、地方性和文化背景的大规模资源。本研究提出了一种名为NativQA的框架，该框架能够无缝构建大规模、文化上和地区上对齐的多语言问答数据集。该框架利用用户定义的种子查询，并利用搜索引擎收集地点特定的日常信息。该框架在24个国家的39个地点进行了评估，涵盖7种从极度低资源到高资源的语言，生成了超过30万条问答对。开发的资源可用于语言模型的评估和进一步微调。该框架已向社区公开（this https URL）。 

---
# Enhancing Coreference Resolution with Pretrained Language Models: Bridging the Gap Between Syntax and Semantics 

**Title (ZH)**: 使用预训练语言模型增强同指替解析：缩小语法与语义之间的差距 

**Authors**: Xingzu Liu, Songhang deng, Mingbang Wang, Zhang Dong, Le Dai, Jiyuan Li, Ruilin Nong  

**Link**: [PDF](https://arxiv.org/pdf/2504.05855)  

**Abstract**: Large language models have made significant advancements in various natural language processing tasks, including coreference resolution. However, traditional methods often fall short in effectively distinguishing referential relationships due to a lack of integration between syntactic and semantic information. This study introduces an innovative framework aimed at enhancing coreference resolution by utilizing pretrained language models. Our approach combines syntax parsing with semantic role labeling to accurately capture finer distinctions in referential relationships. By employing state-of-the-art pretrained models to gather contextual embeddings and applying an attention mechanism for fine-tuning, we improve the performance of coreference tasks. Experimental results across diverse datasets show that our method surpasses conventional coreference resolution systems, achieving notable accuracy in disambiguating references. This development not only improves coreference resolution outcomes but also positively impacts other natural language processing tasks that depend on precise referential understanding. 

**Abstract (ZH)**: 大型语言模型在各种自然语言处理任务中取得了显著进展，包括核心ference解析。然而，传统方法往往因缺乏语法和语义信息的整合而在有效区分指称关系方面力有未逮。本研究提出了一种创新框架，旨在通过利用预训练语言模型来提升核心ference解析。我们的方法结合了句法解析和语义角色标注，以准确捕获指称关系中的细微区别。通过使用最先进的预训练模型来收集上下文嵌入，并应用注意力机制进行微调，我们提高了核心ference任务的性能。跨多个数据集的实验结果表明，我们的方法超越了传统的核心ference解析系统，实现了显著的参考消歧准确性。这一发展不仅提升了核心ference解析的效果，还对依赖精确指称理解的其他自然语言处理任务产生了积极影响。 

---
# PathGPT: Leveraging Large Language Models for Personalized Route Generation 

**Title (ZH)**: PathGPT：利用大型语言模型进行个性化路径生成 

**Authors**: Steeve Cuthbert Marcelyn, Yucen Gao, Yuzhe Zhang, Xiaofeng Gao, Guihai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.05846)  

**Abstract**: The proliferation of GPS enabled devices has led to the accumulation of a substantial corpus of historical trajectory data. By leveraging these data for training machine learning models,researchers have devised novel data-driven methodologies that address the personalized route recommendation (PRR) problem. In contrast to conventional algorithms such as Dijkstra shortest path algorithm,these novel algorithms possess the capacity to discern and learn patterns within the data,thereby facilitating the generation of more personalized paths. However,once these models have been trained,their application is constrained to the generation of routes that align with their training patterns. This limitation renders them less adaptable to novel scenarios and the deployment of multiple machine learning models might be necessary to address new possible scenarios,which can be costly as each model must be trained separately. Inspired by recent advances in the field of Large Language Models (LLMs),we leveraged their natural language understanding capabilities to develop a unified model to solve the PRR problem while being seamlessly adaptable to new scenarios without additional training. To accomplish this,we combined the extensive knowledge LLMs acquired during training with further access to external hand-crafted context information,similar to RAG (Retrieved Augmented Generation) systems,to enhance their ability to generate paths according to user-defined requirements. Extensive experiments on different datasets show a considerable uplift in LLM performance on the PRR problem. 

**Abstract (ZH)**: GPS-enable装置的普及导致了大量历史轨迹数据的积累。通过利用这些数据训练机器学习模型，研究人员设计出了新的数据驱动方法来解决个性化路线推荐问题。与传统的算法（如迪杰斯特拉最短路径算法）相比，这些新型算法能够识别和学习数据中的模式，从而生成更个性化的路径。然而，一旦这些模型被训练好，它们的应用就仅限于生成与其训练模式相符的路线。这一限制使得它们难以适应新的场景，而部署多个机器学习模型以应对新场景可能会非常昂贵，因为每个模型都需要单独训练。受大型语言模型（LLMs）领域近期进展的启发，我们利用了其自然语言理解能力，开发了一个统一的模型来解决个性化路线推荐问题，并使其能够无缝适应新场景而无需额外训练。为此，我们结合了LLMs在训练过程中获得的丰富知识，并进一步提供了手动生成的外部上下文信息，类似于RAG（检索增强生成）系统，以增强其根据用户自定义需求生成路径的能力。在不同数据集上的 extensive 实验显示了 LLM 在个性化路线推荐问题上的显著性能提升。 

---
# How to Enable LLM with 3D Capacity? A Survey of Spatial Reasoning in LLM 

**Title (ZH)**: 如何启用具有三维能力的LLM？关于LLM中空间推理的综述 

**Authors**: Jirong Zha, Yuxuan Fan, Xiao Yang, Chen Gao, Xinlei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.05786)  

**Abstract**: 3D spatial understanding is essential in real-world applications such as robotics, autonomous vehicles, virtual reality, and medical imaging. Recently, Large Language Models (LLMs), having demonstrated remarkable success across various domains, have been leveraged to enhance 3D understanding tasks, showing potential to surpass traditional computer vision methods. In this survey, we present a comprehensive review of methods integrating LLMs with 3D spatial understanding. We propose a taxonomy that categorizes existing methods into three branches: image-based methods deriving 3D understanding from 2D visual data, point cloud-based methods working directly with 3D representations, and hybrid modality-based methods combining multiple data streams. We systematically review representative methods along these categories, covering data representations, architectural modifications, and training strategies that bridge textual and 3D modalities. Finally, we discuss current limitations, including dataset scarcity and computational challenges, while highlighting promising research directions in spatial perception, multi-modal fusion, and real-world applications. 

**Abstract (ZH)**: 三维空间理解在机器人技术、自动驾驶车辆、虚拟现实和医学成像等实际应用中至关重要。近年来，大型语言模型（LLMs）在各个领域展现出显著的成功，并被用于增强三维理解任务，显示出超越传统计算机视觉方法的潜力。在本文综述中，我们对将LLMs与三维空间理解相结合的方法进行了全面回顾。我们提出了一个分类体系，将现有方法归类为三大分支：基于图像的方法、基于点云的方法以及混合模态方法。我们系统地回顾了这些类别中的代表性方法，涵盖数据表示、架构修改以及将文本和三维模态结合起来的训练策略。最后，我们讨论了当前的局限性，包括数据集稀缺性和计算挑战，并强调了在空间感知、多模态融合和实际应用方面的有前景的研究方向。 

---
# MDK12-Bench: A Multi-Discipline Benchmark for Evaluating Reasoning in Multimodal Large Language Models 

**Title (ZH)**: MDK12-Bench: 一种多学科基准，用于评估多模态大型语言模型的推理能力 

**Authors**: Pengfei Zhou, Fanrui Zhang, Xiaopeng Peng, Zhaopan Xu, Jiaxin Ai, Yansheng Qiu, Chuanhao Li, Zhen Li, Ming Li, Yukang Feng, Jianwen Sun, Haoquan Zhang, Zizhen Li, Xiaofeng Mao, Wangbo Zhao, Kai Wang, Xiaojun Chang, Wenqi Shao, Yang You, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05782)  

**Abstract**: Multimodal reasoning, which integrates language and visual cues into problem solving and decision making, is a fundamental aspect of human intelligence and a crucial step toward artificial general intelligence. However, the evaluation of multimodal reasoning capabilities in Multimodal Large Language Models (MLLMs) remains inadequate. Most existing reasoning benchmarks are constrained by limited data size, narrow domain coverage, and unstructured knowledge distribution. To close these gaps, we introduce MDK12-Bench, a multi-disciplinary benchmark assessing the reasoning capabilities of MLLMs via real-world K-12 examinations. Spanning six disciplines (math, physics, chemistry, biology, geography, and information science), our benchmark comprises 140K reasoning instances across diverse difficulty levels from primary school to 12th grade. It features 6,827 instance-level knowledge point annotations based on a well-organized knowledge structure, detailed answer explanations, difficulty labels and cross-year partitions, providing a robust platform for comprehensive evaluation. Additionally, we present a novel dynamic evaluation framework to mitigate data contamination issues by bootstrapping question forms, question types, and image styles during evaluation. Extensive experiment on MDK12-Bench reveals the significant limitation of current MLLMs in multimodal reasoning. The findings on our benchmark provide insights into the development of the next-generation models. Our data and codes are available at this https URL. 

**Abstract (ZH)**: 多模态推理能力评估：MDK12-Bench多学科基准测试 

---
# Rank-Then-Score: Enhancing Large Language Models for Automated Essay Scoring 

**Title (ZH)**: 基于排名然后评分的方法：增强大规模语言模型以实现自动化作文评分 

**Authors**: Yida Cai, Kun Liang, Sanwoo Lee, Qinghan Wang, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05736)  

**Abstract**: In recent years, large language models (LLMs) achieve remarkable success across a variety of tasks. However, their potential in the domain of Automated Essay Scoring (AES) remains largely underexplored. Moreover, compared to English data, the methods for Chinese AES is not well developed. In this paper, we propose Rank-Then-Score (RTS), a fine-tuning framework based on large language models to enhance their essay scoring capabilities. Specifically, we fine-tune the ranking model (Ranker) with feature-enriched data, and then feed the output of the ranking model, in the form of a candidate score set, with the essay content into the scoring model (Scorer) to produce the final score. Experimental results on two benchmark datasets, HSK and ASAP, demonstrate that RTS consistently outperforms the direct prompting (Vanilla) method in terms of average QWK across all LLMs and datasets, and achieves the best performance on Chinese essay scoring using the HSK dataset. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在各种任务中取得了显著成功。然而，它们在自动作文评分（AES）领域的潜力尚未得到充分探索。此外，与英语数据相比，中文AES的方法发展不够成熟。在本文中，我们提出了一种基于大规模语言模型的Rank-Then-Score（RTS）微调框架，以增强其作文评分能力。具体来说，我们使用特征丰富化的数据对排序模型（Ranker）进行微调，然后将排序模型的输出（候选评分集合）与作文内容一起输入评分模型（Scorer），以生成最终评分。在HSK和ASAP两个基准数据集上的实验结果表明，RTS在所有LLM和数据集中平均QWK指标上始终优于直接提示（Vanilla）方法，并在HSK数据集上实现了最佳的中文作文评分性能。 

---
# Large Language Models Enhanced Hyperbolic Space Recommender Systems 

**Title (ZH)**: 大型语言模型增强的双曲空间推荐系统 

**Authors**: Wentao Cheng, Zhida Qin, Zexue Wu, Pengzhan Zhou, Tianyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05694)  

**Abstract**: Large Language Models (LLMs) have attracted significant attention in recommender systems for their excellent world knowledge capabilities. However, existing methods that rely on Euclidean space struggle to capture the rich hierarchical information inherent in textual and semantic data, which is essential for capturing user preferences. The geometric properties of hyperbolic space offer a promising solution to address this issue. Nevertheless, integrating LLMs-based methods with hyperbolic space to effectively extract and incorporate diverse hierarchical information is non-trivial. To this end, we propose a model-agnostic framework, named HyperLLM, which extracts and integrates hierarchical information from both structural and semantic perspectives. Structurally, HyperLLM uses LLMs to generate multi-level classification tags with hierarchical parent-child relationships for each item. Then, tag-item and user-item interactions are jointly learned and aligned through contrastive learning, thereby providing the model with clear hierarchical information. Semantically, HyperLLM introduces a novel meta-optimized strategy to extract hierarchical information from semantic embeddings and bridge the gap between the semantic and collaborative spaces for seamless integration. Extensive experiments show that HyperLLM significantly outperforms recommender systems based on hyperbolic space and LLMs, achieving performance improvements of over 40%. Furthermore, HyperLLM not only improves recommender performance but also enhances training stability, highlighting the critical role of hierarchical information in recommender systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在推荐系统中的世界知识能力使其在推荐领域引起了广泛关注。然而，现有的依赖欧几里得空间的方法难以捕捉文本和语义数据中固有的丰富层次信息，而这些信息对于捕捉用户偏好至关重要。双曲空间的几何性质提供了解决这一问题的潜在方案。然而，将LLMs方法与双曲空间有效结合以提取和整合多种层次信息仍然是一个挑战。为此，我们提出了一种模型通用框架HyperLLM，从结构和语义两个视角提取并整合层次信息。从结构上看，HyperLLM 使用LLMs为每个项目生成具有层次父子女关系的多级分类标签。然后，通过对比学习联合学习和对齐标签-项目和用户-项目的交互，从而为模型提供了清晰的层次信息。从语义上看，HyperLLM 引入了一种新颖的元优化策略，从语义嵌入中提取层次信息，并在语义和协同空间之间建立桥梁，以实现无缝集成。广泛的实验表明，HyperLLM 显著优于基于双曲空间和LLMs的推荐系统，性能提升超过40%。此外，HyperLLM 不仅提高了推荐性能，还增强了训练稳定性，突显了层次信息在推荐系统中的关键作用。 

---
# STRIVE: A Think & Improve Approach with Iterative Refinement for Enhancing Question Quality Estimation 

**Title (ZH)**: STRIVE: 一种迭代优化的思考与改进方法以提升问题质量估测 

**Authors**: Aniket Deroy, Subhankar Maity  

**Link**: [PDF](https://arxiv.org/pdf/2504.05693)  

**Abstract**: Automatically assessing question quality is crucial for educators as it saves time, ensures consistency, and provides immediate feedback for refining teaching materials. We propose a novel methodology called STRIVE (Structured Thinking and Refinement with multiLLMs for Improving Verified Question Estimation) using a series of Large Language Models (LLMs) for automatic question evaluation. This approach aims to improve the accuracy and depth of question quality assessment, ultimately supporting diverse learners and enhancing educational practices. The method estimates question quality in an automated manner by generating multiple evaluations based on the strengths and weaknesses of the provided question and then choosing the best solution generated by the LLM. Then the process is improved by iterative review and response with another LLM until the evaluation metric values converge. This sophisticated method of evaluating question quality improves the estimation of question quality by automating the task of question quality evaluation. Correlation scores show that using this proposed method helps to improve correlation with human judgments compared to the baseline method. Error analysis shows that metrics like relevance and appropriateness improve significantly relative to human judgments by using STRIVE. 

**Abstract (ZH)**: 自动评估问题质量对于教育者至关重要，因为它可以节省时间、确保一致性和提供即时反馈以改进教学材料。我们提出了一种新型方法STRIVE（结构化思考与多LLMs辅助改进验证问题估计），利用一系列大型语言模型（LLMs）进行自动问题评估。该方法旨在提高问题质量评估的准确性和深度，最终支持多样化的学习者并增强教育实践。该方法通过对提供的问题的优点和缺点生成多个评估来自动估计问题质量，然后选择由LLM生成的最佳解决方案。该过程通过与另一个LLM进行迭代审查和响应，直到评估指标值收敛，从而改进了问题质量评估。相关性分析表明，使用该提出的 STRIVE 方法有助于提高与人类判断的相关性，而基准方法则不然。误差分析表明，通过使用 STRIVE，相关度和适宜性等指标相对于人类判断有显著改善。 

---
# Towards Smarter Hiring: Are Zero-Shot and Few-Shot Pre-trained LLMs Ready for HR Spoken Interview Transcript Analysis? 

**Title (ZH)**: 基于零样本和少样本预训练大语言模型，招聘面试转录分析是否 Ready 了？ 

**Authors**: Subhankar Maity, Aniket Deroy, Sudeshna Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2504.05683)  

**Abstract**: This research paper presents a comprehensive analysis of the performance of prominent pre-trained large language models (LLMs), including GPT-4 Turbo, GPT-3.5 Turbo, text-davinci-003, text-babbage-001, text-curie-001, text-ada-001, llama-2-7b-chat, llama-2-13b-chat, and llama-2-70b-chat, in comparison to expert human evaluators in providing scores, identifying errors, and offering feedback and improvement suggestions to candidates during mock HR (Human Resources) interviews. We introduce a dataset called HURIT (Human Resource Interview Transcripts), which comprises 3,890 HR interview transcripts sourced from real-world HR interview scenarios. Our findings reveal that pre-trained LLMs, particularly GPT-4 Turbo and GPT-3.5 Turbo, exhibit commendable performance and are capable of producing evaluations comparable to those of expert human evaluators. Although these LLMs demonstrate proficiency in providing scores comparable to human experts in terms of human evaluation metrics, they frequently fail to identify errors and offer specific actionable advice for candidate performance improvement in HR interviews. Our research suggests that the current state-of-the-art pre-trained LLMs are not fully conducive for automatic deployment in an HR interview assessment. Instead, our findings advocate for a human-in-the-loop approach, to incorporate manual checks for inconsistencies and provisions for improving feedback quality as a more suitable strategy. 

**Abstract (ZH)**: 本研究论文对包括GPT-4 Turbo、GPT-3.5 Turbo、text-davinci-003、text-babbage-001、text-curie-001、text-ada-001、llama-2-7b-chat、llama-2-13b-chat和llama-2-70b-chat等 prominent 预训练大规模语言模型（LLMs）在模拟人力资源（HR）面试中提供评分、识别错误和提供反馈及改进建议方面的性能进行了全面分析，并将其与专家人力资源评估员进行了比较。我们介绍了名为HURIT（人力资源面试转录）的数据集，包含3,890份实际人力资源面试场景的转录。我们的研究发现，预训练LLMs，特别是GPT-4 Turbo和GPT-3.5 Turbo表现出色，能够产生与专家人力资源评估员相当的评价。尽管这些LLMs在提供评分方面在人力资源评估指标中表现出与人类专家相当的熟练度，但在识别错误和为候选人面试表现提供具体可操作的改进建议方面却经常失败。研究结果表明，当前最先进的预训练LLMs并不完全适合在人力资源面试评估中的自动部署。相反，我们的研究建议采用人工在环中的方法，结合人工检查不一致性和提供提高反馈质量的方法，作为更合适的策略。 

---
# Reasoning Towards Fairness: Mitigating Bias in Language Models through Reasoning-Guided Fine-Tuning 

**Title (ZH)**: 基于推理的公平性推理：通过推理指导微调减轻语言模型中的偏见 

**Authors**: Sanchit Kabra, Akshita Jha, Chandan Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2504.05632)  

**Abstract**: Recent advances in large-scale generative language models have shown that reasoning capabilities can significantly improve model performance across a variety of tasks. However, the impact of reasoning on a model's ability to mitigate stereotypical responses remains largely underexplored. In this work, we investigate the crucial relationship between a model's reasoning ability and fairness, and ask whether improved reasoning capabilities can mitigate harmful stereotypical responses, especially those arising due to shallow or flawed reasoning. We conduct a comprehensive evaluation of multiple open-source LLMs, and find that larger models with stronger reasoning abilities exhibit substantially lower stereotypical bias on existing fairness benchmarks. Building on this insight, we introduce ReGiFT -- Reasoning Guided Fine-Tuning, a novel approach that extracts structured reasoning traces from advanced reasoning models and infuses them into models that lack such capabilities. We use only general-purpose reasoning and do not require any fairness-specific supervision for bias mitigation. Notably, we see that models fine-tuned using ReGiFT not only improve fairness relative to their non-reasoning counterparts but also outperform advanced reasoning models on fairness benchmarks. We also analyze how variations in the correctness of the reasoning traces and their length influence model fairness and their overall performance. Our findings highlight that enhancing reasoning capabilities is an effective, fairness-agnostic strategy for mitigating stereotypical bias caused by reasoning flaws. 

**Abstract (ZH)**: 近年来，大规模生成语言模型的研究进展表明，推理能力可以显著提高模型在各种任务中的性能。然而，推理对模型减轻刻板印象响应能力的影响仍然很大程度上未被探索。在本文中，我们研究了模型的推理能力与其公平性之间的关键关系，并询问改进的推理能力是否能够减轻有害的刻板印象响应，尤其是那些由于浅薄或有缺陷的推理引起的响应。我们对多个开源大语言模型进行了全面评估，发现具有更强推理能力的较大模型在现有的公平性基准测试中表现出显著较低的刻板印象偏见。基于这一洞察，我们提出了一种名为ReGiFT（Reasoning Guided Fine-Tuning）的新方法，该方法从先进的推理模型中提取结构化的推理轨迹，并将它们注入缺乏此类能力的模型中。我们仅使用通用推理，无需任何针对公平性的特定监督即可减轻偏见。值得注意的是，使用ReGiFT进行微调后的模型不仅相对于其非推理版本提高了公平性，还在公平性基准测试中优于先进的推理模型。我们还分析了推理轨迹正确性和长度的变化如何影响模型的公平性和整体性能。我们的发现强调，增强推理能力是一种公平性无关的有效策略，用于减轻由于推理缺陷引起的刻板印象偏见。 

---
# FactGuard: Leveraging Multi-Agent Systems to Generate Answerable and Unanswerable Questions for Enhanced Long-Context LLM Extraction 

**Title (ZH)**: FactGuard：利用多智能体系统生成可回答和不可回答的问题以增强长上下文语言模型提取 

**Authors**: Qian-Wen Zhang, Fang Li, Jie Wang, Lingfeng Qiao, Yifei Yu, Di Yin, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.05607)  

**Abstract**: Extractive reading comprehension systems are designed to locate the correct answer to a question within a given text. However, a persistent challenge lies in ensuring these models maintain high accuracy in answering questions while reliably recognizing unanswerable queries. Despite significant advances in large language models (LLMs) for reading comprehension, this issue remains critical, particularly as the length of supported contexts continues to expand. To address this challenge, we propose an innovative data augmentation methodology grounded in a multi-agent collaborative framework. Unlike traditional methods, such as the costly human annotation process required for datasets like SQuAD 2.0, our method autonomously generates evidence-based question-answer pairs and systematically constructs unanswerable questions. Using this methodology, we developed the FactGuard-Bench dataset, which comprises 25,220 examples of both answerable and unanswerable question scenarios, with context lengths ranging from 8K to 128K. Experimental evaluations conducted on seven popular LLMs reveal that even the most advanced models achieve only 61.79% overall accuracy. Furthermore, we emphasize the importance of a model's ability to reason about unanswerable questions to avoid generating plausible but incorrect answers. By implementing efficient data selection and generation within the multi-agent collaborative framework, our method significantly reduces the traditionally high costs associated with manual annotation and provides valuable insights for the training and optimization of LLMs. 

**Abstract (ZH)**: 基于多Agent协作框架的数据增强方法：提高不可答查询识别的抽取式阅读理解系统 

---
# Knowledge-Instruct: Effective Continual Pre-training from Limited Data using Instructions 

**Title (ZH)**: Knowledge-Instruct：使用指令从有限数据中进行有效的持续预训练 

**Authors**: Oded Ovadia, Meni Brief, Rachel Lemberg, Eitam Sheetrit  

**Link**: [PDF](https://arxiv.org/pdf/2504.05571)  

**Abstract**: While Large Language Models (LLMs) acquire vast knowledge during pre-training, they often lack domain-specific, new, or niche information. Continual pre-training (CPT) attempts to address this gap but suffers from catastrophic forgetting and inefficiencies in low-data regimes. We introduce Knowledge-Instruct, a novel approach to efficiently inject knowledge from limited corpora through pure instruction-tuning. By generating information-dense synthetic instruction data, it effectively integrates new knowledge while preserving general reasoning and instruction-following abilities. Knowledge-Instruct demonstrates superior factual memorization, minimizes catastrophic forgetting, and remains scalable by leveraging synthetic data from relatively small language models. Additionally, it enhances contextual understanding, including complex multi-hop reasoning, facilitating integration with retrieval systems. We validate its effectiveness across diverse benchmarks, including Companies, a new dataset that we release to measure knowledge injection capabilities. 

**Abstract (ZH)**: Knowledge-Instruct：通过纯粹的指令调优高效注入有限语料库中的知识 

---
# Bridging Industrial Expertise and XR with LLM-Powered Conversational Agents 

**Title (ZH)**: 工业 expertise 与 XR 的桥梁：基于 LLM 的对话代理 

**Authors**: Despina Tomkou, George Fatouros, Andreas Andreou, Georgios Makridis, Fotis Liarokapis, Dimitrios Dardanis, Athanasios Kiourtis, John Soldatos, Dimosthenis Kyriazis  

**Link**: [PDF](https://arxiv.org/pdf/2504.05527)  

**Abstract**: This paper introduces a novel integration of Retrieval-Augmented Generation (RAG) enhanced Large Language Models (LLMs) with Extended Reality (XR) technologies to address knowledge transfer challenges in industrial environments. The proposed system embeds domain-specific industrial knowledge into XR environments through a natural language interface, enabling hands-free, context-aware expert guidance for workers. We present the architecture of the proposed system consisting of an LLM Chat Engine with dynamic tool orchestration and an XR application featuring voice-driven interaction. Performance evaluation of various chunking strategies, embedding models, and vector databases reveals that semantic chunking, balanced embedding models, and efficient vector stores deliver optimal performance for industrial knowledge retrieval. The system's potential is demonstrated through early implementation in multiple industrial use cases, including robotic assembly, smart infrastructure maintenance, and aerospace component servicing. Results indicate potential for enhancing training efficiency, remote assistance capabilities, and operational guidance in alignment with Industry 5.0's human-centric and resilient approach to industrial development. 

**Abstract (ZH)**: 本文介绍了一种将增强型大型语言模型（LLMs）与检索增强生成（RAG）技术结合拓展现实（XR）技术的方法，以解决工业环境中的知识转移挑战。提出的系统通过自然语言接口将特定领域的工业知识嵌入XR环境中，为工人提供免手持、情境感知的专业指导。本文介绍了一种LLM聊天引擎架构，其中包含动态工具编排，并展示了一款以其为基础的具备语音驱动交互功能的XR应用程序。各种分块策略、嵌入模型和向量数据库的性能评估表明，语义分块、均衡嵌入模型和高效向量存储为工业知识检索提供了最优性能。通过在多种工业应用场景中的早期实施，包括机器人装配、智能基础设施维护和航空部件服务，展示了该系统的潜力。结果表明，该系统有助于提高培训效率、远程协助能力，并与工业5.0以人类为中心、具备弹性的工业发展方法保持一致。 

---
# Fast Controlled Generation from Language Models with Adaptive Weighted Rejection Sampling 

**Title (ZH)**: 基于自适应加权拒绝采样的语言模型快速受控生成 

**Authors**: Benjamin Lipkin, Benjamin LeBrun, Jacob Hoover Vigly, João Loula, David R. MacIver, Li Du, Jason Eisner, Ryan Cotterell, Vikash Mansinghka, Timothy J. O'Donnell, Alexander K. Lew, Tim Vieira  

**Link**: [PDF](https://arxiv.org/pdf/2504.05410)  

**Abstract**: The dominant approach to generating from language models subject to some constraint is locally constrained decoding (LCD), incrementally sampling tokens at each time step such that the constraint is never violated. Typically, this is achieved through token masking: looping over the vocabulary and excluding non-conforming tokens. There are two important problems with this approach. (i) Evaluating the constraint on every token can be prohibitively expensive -- LM vocabularies often exceed $100,000$ tokens. (ii) LCD can distort the global distribution over strings, sampling tokens based only on local information, even if they lead down dead-end paths. This work introduces a new algorithm that addresses both these problems. First, to avoid evaluating a constraint on the full vocabulary at each step of generation, we propose an adaptive rejection sampling algorithm that typically requires orders of magnitude fewer constraint evaluations. Second, we show how this algorithm can be extended to produce low-variance, unbiased estimates of importance weights at a very small additional cost -- estimates that can be soundly used within previously proposed sequential Monte Carlo algorithms to correct for the myopic behavior of local constraint enforcement. Through extensive empirical evaluation in text-to-SQL, molecular synthesis, goal inference, pattern matching, and JSON domains, we show that our approach is superior to state-of-the-art baselines, supporting a broader class of constraints and improving both runtime and performance. Additional theoretical and empirical analyses show that our method's runtime efficiency is driven by its dynamic use of computation, scaling with the divergence between the unconstrained and constrained LM, and as a consequence, runtime improvements are greater for better models. 

**Abstract (ZH)**: 生成受约束的语言模型输出的新算法：解决局部约束解码的关键问题 

---
# Debate-Feedback: A Multi-Agent Framework for Efficient Legal Judgment Prediction 

**Title (ZH)**: 辩论-反馈：一种高效的法律判决预测多agent框架 

**Authors**: Xi Chen, Mao Mao, Shuo Li, Haotian Shangguan  

**Link**: [PDF](https://arxiv.org/pdf/2504.05358)  

**Abstract**: The use of AI in legal analysis and prediction (LegalAI) has gained widespread attention, with past research focusing on retrieval-based methods and fine-tuning large models. However, these approaches often require large datasets and underutilize the capabilities of modern large language models (LLMs). In this paper, inspired by the debate phase of real courtroom trials, we propose a novel legal judgment prediction model based on the Debate-Feedback architecture, which integrates LLM multi-agent debate and reliability evaluation models. Unlike traditional methods, our model achieves significant improvements in efficiency by minimizing the need for large historical datasets, thus offering a lightweight yet robust solution. Comparative experiments show that it outperforms several general-purpose and domain-specific legal models, offering a dynamic reasoning process and a promising direction for future LegalAI research. 

**Abstract (ZH)**: AI在法律分析与预测中的应用（LegalAI）：基于 Debate-Feedback 架构的新型法律判决预测模型及其优势 

---
# Achieving binary weight and activation for LLMs using Post-Training Quantization 

**Title (ZH)**: 使用后训练量化实现LLMs的二值权重和激活 

**Authors**: Siqing Song, Chuang Wang, Ruiqi Wang, Yi Yang, Xuyao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.05352)  

**Abstract**: Quantizing large language models (LLMs) to 1-bit precision significantly reduces computational costs, but existing quantization techniques suffer from noticeable performance degradation when using weight and activation precisions below 4 bits (W4A4). In this paper, we propose a post-training quantization framework with W(1+1)A(1*4) configuration, where weights are quantized to 1 bit with an additional 1 bit for fine-grain grouping and activations are quantized to 1 bit with a 4-fold increase in the number of channels. For weight quantization, we propose utilizing Hessian-aware fine-grained grouping along with an EM-based quantization scheme. For activation quantization, we decompose INT4-quantized activations into a 4 * INT1 format equivalently and simultaneously smooth the scaling factors based on quantization errors, which further reduces the quantization errors in activations. Our method surpasses state-of-the-art (SOTA) LLM quantization baselines on W2A4 across multiple tasks, pushing the boundaries of existing LLM quantization methods toward fully binarized models. 

**Abstract (ZH)**: 将大型语言模型（LLMs）量化到1比特精度显著降低了计算成本，但现有的量化技术在使用低于4比特的权重和激活精度（W4A4）时会遭受明显的性能下降。本文提出了一种后训练量化框架，配置为W(1+1)A(1*4)，其中权重被量化为1比特，并附加1比特用于细粒度分组，激活值被量化为1比特，通道数增加4倍。在权重量化方面，我们提出了利用海森矩阵感知的细粒度分组结合EM基于的量化方案。在激活量化方面，我们将INT4量化激活等效分解为4 * INT1格式，并同时根据量化误差平滑缩放因子，进一步减少激活的量化误差。我们的方法在多个任务上超越了W2A4的最新基准量化方法，推动了现有LLM量化方法向全二值化模型的边界。 

---
# Thanos: A Block-wise Pruning Algorithm for Efficient Large Language Model Compression 

**Title (ZH)**: Thanos：一种块级裁剪算法，用于高效的大语言模型压缩 

**Authors**: Ivan Ilin, Peter Richtarik  

**Link**: [PDF](https://arxiv.org/pdf/2504.05346)  

**Abstract**: This paper presents Thanos, a novel weight-pruning algorithm designed to reduce the memory footprint and enhance the computational efficiency of large language models (LLMs) by removing redundant weights while maintaining accuracy. Thanos introduces a block-wise pruning strategy with adaptive masks that dynamically adjust to weight importance, enabling flexible sparsity patterns and structured formats, such as $n:m$ sparsity, optimized for hardware acceleration. Experimental evaluations demonstrate that Thanos achieves state-of-the-art performance in structured pruning and outperforms existing methods in unstructured pruning. By providing an efficient and adaptable approach to model compression, Thanos offers a practical solution for deploying large models in resource-constrained environments. 

**Abstract (ZH)**: Thanos：一种新型的权重剪枝算法，通过去除冗余权重以减少大语言模型的内存占用并提高计算效率，同时保持准确性 

---
# AROMA: Autonomous Rank-one Matrix Adaptation 

**Title (ZH)**: AROMA：自主秩一矩阵适应 

**Authors**: Hao Nan Sheng, Zhi-yong Wang, Mingrui Yang, Hing Cheung So  

**Link**: [PDF](https://arxiv.org/pdf/2504.05343)  

**Abstract**: As large language models continue to grow in size, parameter-efficient fine-tuning has become increasingly crucial. While low-rank adaptation (LoRA) offers a solution through low-rank updates, its static rank allocation may yield suboptimal results. Adaptive low-rank adaptation (AdaLoRA) improves this with dynamic allocation but remains sensitive to initial and target rank configurations. We introduce AROMA, a framework that automatically constructs layer-specific updates by iteratively building up rank-one components with very few trainable parameters that gradually diminish to zero. Unlike existing methods that employ rank reduction mechanisms, AROMA introduces a dual-loop architecture for rank growth. The inner loop extracts information from each rank-one subspace, while the outer loop determines the number of rank-one subspaces, i.e., the optimal rank. We reset optimizer states to maintain subspace independence. AROMA significantly reduces parameters compared to LoRA and AdaLoRA while achieving superior performance on natural language understanding and commonsense reasoning tasks, offering new insights into adaptive parameter-efficient fine-tuning. The code is available at \href{this https URL}{AROMA}. 

**Abstract (ZH)**: 随着大型语言模型的不断扩大，参数高效的微调变得越来越重要。虽然低秩适应（LoRA）通过低秩更新提供了解决方案，但其静态秩分配可能导致次优结果。自适应低秩适应（AdaLoRA）通过动态分配改进了这一问题，但仍对初始和目标秩配置敏感。我们提出了AROMA框架，该框架通过迭代构建具有极少可训练参数的秩一子空间，并逐渐减少到零，自动构建层特定的更新。AROMA不同于现有方法使用秩减少机制，引入了双环架构以促进秩增长。内环从每个秩一子空间中提取信息，外环确定秩一子空间的数量，即最优秩。我们重置优化器状态以保持子空间独立性。AROMA在参数量显著少于LoRA和AdaLoRA的情况下，实现了自然语言理解和常识推理任务上的优越性能，为自适应参数高效微调提供了新的见解。代码可在AROMA：[点击这里](this https URL)获取。 

---
# Unequal Opportunities: Examining the Bias in Geographical Recommendations by Large Language Models 

**Title (ZH)**: 机会不均等：考察大型语言模型在地理位置推荐中的偏见 

**Authors**: Shiran Dudy, Thulasi Tholeti, Resmi Ramachandranpillai, Muhammad Ali, Toby Jia-Jun Li, Ricardo Baeza-Yates  

**Link**: [PDF](https://arxiv.org/pdf/2504.05325)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have made them a popular information-seeking tool among end users. However, the statistical training methods for LLMs have raised concerns about their representation of under-represented topics, potentially leading to biases that could influence real-world decisions and opportunities. These biases could have significant economic, social, and cultural impacts as LLMs become more prevalent, whether through direct interactions--such as when users engage with chatbots or automated assistants--or through their integration into third-party applications (as agents), where the models influence decision-making processes and functionalities behind the scenes. Our study examines the biases present in LLMs recommendations of U.S. cities and towns across three domains: relocation, tourism, and starting a business. We explore two key research questions: (i) How similar LLMs responses are, and (ii) How this similarity might favor areas with certain characteristics over others, introducing biases. We focus on the consistency of LLMs responses and their tendency to over-represent or under-represent specific locations. Our findings point to consistent demographic biases in these recommendations, which could perpetuate a ``rich-get-richer'' effect that widens existing economic disparities. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的进步使其成为终端用户常用的检索信息工具。然而，LLMs的统计训练方法引发了对其对代表性不足主题的表示的担忧，可能导致偏见，进而影响现实世界的决策和机会。随着LLMs的应用越来越广泛，无论通过直接交互（如用户与聊天机器人或自动化助手互动）还是通过其嵌入第三方应用程序（作为代理），这些偏见都可能产生重大的经济、社会和文化影响。我们的研究考察了LLMs在美国城市和城镇推荐中的偏见，涵盖了三个领域：迁移、旅游和创业。我们探讨了两个核心研究问题：（i）LLMs响应的相似性，以及（ii）这种相似性如何可能有利于某些特征的地区，引入偏见。我们关注LLMs响应的一致性和其过度代表或低估特定地点的倾向。研究发现这些推荐中存在一致的人口统计学偏见，这可能会加剧现有的经济不平等。 

---
# Hybrid Retrieval for Hallucination Mitigation in Large Language Models: A Comparative Analysis 

**Title (ZH)**: 大型语言模型中幻觉缓解的混合检索方法：一种比较分析 

**Authors**: Chandana Sree Mala, Gizem Gezici, Fosca Giannotti  

**Link**: [PDF](https://arxiv.org/pdf/2504.05324)  

**Abstract**: Large Language Models (LLMs) excel in language comprehension and generation but are prone to hallucinations, producing factually incorrect or unsupported outputs. Retrieval Augmented Generation (RAG) systems address this issue by grounding LLM responses with external knowledge. This study evaluates the relationship between retriever effectiveness and hallucination reduction in LLMs using three retrieval approaches: sparse retrieval based on BM25 keyword search, dense retrieval using semantic search with Sentence Transformers, and a proposed hybrid retrieval module. The hybrid module incorporates query expansion and combines the results of sparse and dense retrievers through a dynamically weighted Reciprocal Rank Fusion score. Using the HaluBench dataset, a benchmark for hallucinations in question answering tasks, we assess retrieval performance with metrics such as mean average precision and normalised discounted cumulative gain, focusing on the relevance of the top three retrieved documents. Results show that the hybrid retriever achieves better relevance scores, outperforming both sparse and dense retrievers. Further evaluation of LLM-generated answers against ground truth using metrics such as accuracy, hallucination rate, and rejection rate reveals that the hybrid retriever achieves the highest accuracy on fails, the lowest hallucination rate, and the lowest rejection rate. These findings highlight the hybrid retriever's ability to enhance retrieval relevance, reduce hallucination rates, and improve LLM reliability, emphasising the importance of advanced retrieval techniques in mitigating hallucinations and improving response accuracy. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在语言理解和生成方面表现出色，但容易产生幻觉，生成事实错误或缺乏支持的输出。检索增强生成（RAG）系统通过将LLM响应与外部知识相结合来解决这一问题。本研究使用三种检索方法评估检索器有效性与幻觉减少之间的关系：基于BM25关键词搜索的稀疏检索、使用Sentence Transformers进行语义搜索的密集检索，以及一个提出的混合检索模块。混合模块结合了查询扩展，并通过动态加权的互逆排名融合得分将稀疏检索和密集检索的结果结合起来。使用HaluBench数据集，这是一个用于问答任务中幻觉的基准测试集，我们使用平均精度均值和归一化累积增益等指标评估检索性能，重点关注检索的前三份文档的相关性。结果表明，混合检索器在相关性评分方面表现更好，优于稀疏和密集检索器。进一步使用准确率、幻觉率和拒绝率等指标评估LLM生成的答案与真实答案的差异，结果显示混合检索器在错误上的准确率最高、幻觉率最低、拒绝率最低。这些发现突显了混合检索器增强检索相关性、降低幻觉率和提高LLM可靠性的能力，强调了先进检索技术在减轻幻觉和提高响应准确性方面的重要性。 

---
# VALUE: Value-Aware Large Language Model for Query Rewriting via Weighted Trie in Sponsored Search 

**Title (ZH)**: 价值感知型大型语言模型：基于加权trie树的赞助搜索查询重写 

**Authors**: Boyang Zuo, Xiao Zhang, Feng Li, Pengjie Wang, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.05321)  

**Abstract**: In the realm of sponsored search advertising, matching advertisements with the search intent of a user's query is crucial. Query-to-bidwords(i.e. bidding keywords) rewriting is a vital technique that has garnered significant attention. Recently, with the prevalence of LLMs, generative retrieval methods have proven effective in producing high-relevance rewrites. However, we have identified a significant limitation in existing approaches: While fine-tuning LLMs for specific domains enhances semantic relevance, these models have no perception of the intrinsic value of their generated outputs, such as commercial value. Therefore, after SFT, a RLHF phase is often employed to address this issue. Nevertheless, traditional preference alignment methods often face challenges in aligning fine-grained values and are susceptible to overfitting, which diminishes the effectiveness and quality of the generated results. To address these challenges, we propose VALUE(Value-Aware Large language model for qUery rewriting via wEighted trie), the first framework that ensures the generation of high-value and highly relevant bidwords. Our approach utilizes weighted trie, an innovative modification of the traditional trie data structure. By modulating the LLM's output probability distribution with value information from the trie during decoding process, we constrain the generation space and guide the trajectory of text production. Offline experiments demonstrate the effectiveness of our method in semantic matching and preference alignment, showing a remarkable improvement in the value attribute by more than fivefold. Online A/B tests further revealed that our Revenue Per Mille (RPM) metric increased by 1.64%. VALUE has been deployed on our advertising system since October 2024 and served the Double Eleven promotions, the biggest shopping carnival in China. 

**Abstract (ZH)**: 基于赞助搜索广告的查询到出价词重写：考虑价值的价值感知大规模语言模型trie框架 

---
# On Synthesizing Data for Context Attribution in Question Answering 

**Title (ZH)**: 基于上下文归因的问答数据合成 

**Authors**: Gorjan Radevski, Kiril Gashteovski, Shahbaz Syed, Christopher Malon, Sebastien Nicolas, Chia-Chien Hung, Timo Sztyler, Verena Heußer, Wiem Ben Rim, Masafumi Enomoto, Kunihiro Takeoka, Masafumi Oyamada, Goran Glavaš, Carolin Lawrence  

**Link**: [PDF](https://arxiv.org/pdf/2504.05317)  

**Abstract**: Question Answering (QA) accounts for a significant portion of LLM usage "in the wild". However, LLMs sometimes produce false or misleading responses, also known as "hallucinations". Therefore, grounding the generated answers in contextually provided information -- i.e., providing evidence for the generated text -- is paramount for LLMs' trustworthiness. Providing this information is the task of context attribution. In this paper, we systematically study LLM-based approaches for this task, namely we investigate (i) zero-shot inference, (ii) LLM ensembling, and (iii) fine-tuning of small LMs on synthetic data generated by larger LLMs. Our key contribution is SynQA: a novel generative strategy for synthesizing context attribution data. Given selected context sentences, an LLM generates QA pairs that are supported by these sentences. This leverages LLMs' natural strengths in text generation while ensuring clear attribution paths in the synthetic training data. We show that the attribution data synthesized via SynQA is highly effective for fine-tuning small LMs for context attribution in different QA tasks and domains. Finally, with a user study, we validate the usefulness of small LMs (fine-tuned on synthetic data from SynQA) in context attribution for QA. 

**Abstract (ZH)**: 基于LLM的方法在问答任务中合成上下文归因数据的研究 

---
# IterQR: An Iterative Framework for LLM-based Query Rewrite in e-Commercial Search System 

**Title (ZH)**: IterQR：基于LLM的电子商务搜索系统中查询重写的一种迭代框架 

**Authors**: Shangyu Chen, Xinyu Jia, Yingfei Zhang, Shuai Zhang, Xiang Li, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.05309)  

**Abstract**: The essence of modern e-Commercial search system lies in matching user's intent and available candidates depending on user's query, providing personalized and precise service. However, user's query may be incorrect due to ambiguous input and typo, leading to inaccurate search. These cases may be released by query rewrite: modify query to other representation or expansion. However, traditional query rewrite replies on static rewrite vocabulary, which is manually established meanwhile lacks interaction with both domain knowledge in e-Commercial system and common knowledge in the real world. In this paper, with the ability to generate text content of Large Language Models (LLMs), we provide an iterative framework to generate query rewrite. The framework incorporates a 3-stage procedure in each iteration: Rewrite Generation with domain knowledge by Retrieval-Augmented Generation (RAG) and query understanding by Chain-of-Thoughts (CoT); Online Signal Collection with automatic positive rewrite update; Post-training of LLM with multi task objective to generate new rewrites. Our work (named as IterQR) provides a comprehensive framework to generate \textbf{Q}uery \textbf{R}ewrite with both domain / real-world knowledge. It automatically update and self-correct the rewrites during \textbf{iter}ations. \method{} has been deployed in Meituan Delivery's search system (China's leading food delivery platform), providing service for users with significant improvement. 

**Abstract (ZH)**: 现代电子商务搜索系统的核心在于根据用户的查询匹配用户的意图和可用的候选对象，提供个性化和精准的服务。然而，由于输入模糊和拼写错误，用户的查询可能会不准确，导致搜索结果不准确。这些情况可以通过查询重写来缓解：即将查询修改为其他表示形式或扩展。然而，传统的查询重写依赖于静态的重写词汇表，该词汇表是手动建立的，并且缺乏与电子商务系统领域知识和现实世界中的普通知识的交互。本文利用大规模语言模型（LLMs）生成文本内容的能力，提供了一个迭代框架来生成查询重写。该框架在每次迭代中包含三个阶段的过程：通过检索增强生成（RAG）和基于链式思考（CoT）的查询理解来进行重写生成；自动收集在线信号以更新正向重写；以及使用多任务目标对LLMs进行后训练以生成新的重写。我们的工作（命名为IterQR）提供了一个全面的框架，结合了领域知识和现实世界知识来生成查询重写。该框架在迭代过程中自动更新和自我校正重写。我们在美团配送的搜索系统（中国领先的食品配送平台）中部署了该方法，为用户提供显著改进的服务。 

---
# When Reasoning Meets Compression: Benchmarking Compressed Large Reasoning Models on Complex Reasoning Tasks 

**Title (ZH)**: 当推理遇到压缩：压缩大型推理模型在复杂推理任务上的基准测试 

**Authors**: Nan Zhang, Yusen Zhang, Prasenjit Mitra, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02010)  

**Abstract**: Recent open-source large reasoning models (LRMs) exhibit strong performance on complex reasoning tasks, but their large parameter count makes them prohibitively expensive for individuals. The compression of large language models (LLMs) offers an effective solution to reduce cost of computational resources. However, systematic studies on the performance of compressed LLMs in complex reasoning tasks, especially for LRMs, are lacking. Most works on quantization and pruning focus on preserving language modeling performance, while existing distillation works do not comprehensively benchmark student models based on reasoning difficulty or compression impact on knowledge and reasoning. In this paper, we benchmark compressed DeepSeek-R1 models on four different reasoning datasets (AIME 2024, FOLIO, Temporal Sequences of BIG-Bench Hard, and MuSiQue), ranging from mathematical to multihop reasoning, using quantization, distillation, and pruning methods. We benchmark 2.51-, 1.73-, and 1.58-bit R1 models that adopt dynamic quantization. We also benchmark distilled R1 models that are based on LLaMA or Qwen and run SparseGPT on them to obtain various sparsity levels. Studying the performance and behavior of compressed LRMs, we report their performance scores and test-time compute (number of tokens spent on each question). Notably, using MuSiQue, we find that parameter count has a much greater impact on LRMs' knowledge memorization than on their reasoning capability, which can inform the choice of compression techniques. Through our empirical analysis of test-time compute, we find that shorter model outputs generally achieve better performance than longer ones across several benchmarks for both R1 and its compressed variants, highlighting the need for more concise reasoning chains. 

**Abstract (ZH)**: Recent开源大型推理模型在复杂推理任务中表现出色，但其庞大的参数量使个人使用成本高昂。大型语言模型的压缩为降低计算资源成本提供了有效解决方案。然而，针对压缩大型语言模型在复杂推理任务中的性能研究，特别是针对推理型模型的研究仍然不足。大部分关于量化和剪枝的工作侧重于保留语言建模性能，而现有的蒸馏工作并没有全面基于推理难度或压缩对知识和推理的影响来评估学生模型。在本文中，我们使用量化、蒸馏和剪枝方法，在四个不同推理数据集（AIME 2024、FOLIO、BIG-Bench Hard中的多跳推理和MuSiQue）上对DeepSeek-R1压缩模型进行基准测试，涵盖从数学到多跳推理的任务。我们还测试了基于LLaMA或Qwen的蒸馏R1模型，并使用SparseGPT获得不同稀疏程度。通过研究压缩推理模型的性能和行为，我们报告了它们的性能分数和测试时间计算（每个问题消耗的token数）。值得注意的是，使用MuSiQue，我们发现参数量对推理模型的知识记忆影响更大，而对推理能力的影响相对较小，这可以指导压缩技术的选择。通过实证分析测试时间计算量，我们发现对于R1及其压缩变体，在多个基准上较短的模型输出通常比较长的模型输出具有更好的性能，强调了更简洁推理链的需求。 

---
