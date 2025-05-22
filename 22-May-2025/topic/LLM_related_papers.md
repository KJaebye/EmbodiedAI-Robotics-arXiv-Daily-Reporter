# HCRMP: A LLM-Hinted Contextual Reinforcement Learning Framework for Autonomous Driving 

**Title (ZH)**: HCRMP：一种基于LLM的上下文强化学习自主驾驶框架 

**Authors**: Zhiwen Chen, Bo Leng, Zhuoren Li, Hanming Deng, Guizhe Jin, Ran Yu, Huanxi Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.15793)  

**Abstract**: Integrating Large Language Models (LLMs) with Reinforcement Learning (RL) can enhance autonomous driving (AD) performance in complex scenarios. However, current LLM-Dominated RL methods over-rely on LLM outputs, which are prone to this http URL show that state-of-the-art LLM indicates a non-hallucination rate of only approximately 57.95% when assessed on essential driving-related tasks. Thus, in these methods, hallucinations from the LLM can directly jeopardize the performance of driving policies. This paper argues that maintaining relative independence between the LLM and the RL is vital for solving the hallucinations problem. Consequently, this paper is devoted to propose a novel LLM-Hinted RL paradigm. The LLM is used to generate semantic hints for state augmentation and policy optimization to assist RL agent in motion planning, while the RL agent counteracts potential erroneous semantic indications through policy learning to achieve excellent driving performance. Based on this paradigm, we propose the HCRMP (LLM-Hinted Contextual Reinforcement Learning Motion Planner) architecture, which is designed that includes Augmented Semantic Representation Module to extend state space. Contextual Stability Anchor Module enhances the reliability of multi-critic weight hints by utilizing information from the knowledge base. Semantic Cache Module is employed to seamlessly integrate LLM low-frequency guidance with RL high-frequency control. Extensive experiments in CARLA validate HCRMP's strong overall driving performance. HCRMP achieves a task success rate of up to 80.3% under diverse driving conditions with different traffic densities. Under safety-critical driving conditions, HCRMP significantly reduces the collision rate by 11.4%, which effectively improves the driving performance in complex scenarios. 

**Abstract (ZH)**: 将大型语言模型（LLM）与强化学习（RL）结合可以增强自动驾驶（AD）在复杂场景中的性能。然而，当前的LLM主导的RL方法过度依赖LLM的输出，这些输出易产生幻觉。研究表明，最先进的LLM在关键驾驶任务上的无幻觉率仅为约57.95%。因此，在这些方法中，LLM的幻觉会直接危及驾驶策略的性能。本文论点认为，保持LLM与RL相对独立对于解决幻觉问题至关重要。因此，本文提出了一种新颖的LLM提示RL范式。LLM用于生成语义提示，以增强状态并优化策略，辅助RL代理进行运动规划，而RL代理则通过策略学习抵消潜在的错误语义指示，以实现优秀的驾驶性能。基于这一范式，我们提出了HCRMP（LLM提示的上下文强化学习运动规划器）架构，该架构设计包括扩展状态空间的增强语义表示模块，利用知识库信息增强多批评注权重提示的可靠性，并通过语义缓存模块无缝集成LLM低频指导与RL高频控制。在CARLA中进行的大量实验验证了HCRMP在各种驾驶条件下的强大整体驾驶性能。在不同交通密度的多种驾驶条件下，HCRMP实现了高达80.3%的任务成功率。在关键驾驶条件下，HCRMP显著降低了碰撞率11.4%，从而在复杂场景中有效地提升了驾驶性能。 

---
# ClickSight: Interpreting Student Clickstreams to Reveal Insights on Learning Strategies via LLMs 

**Title (ZH)**: ClickSight: 通过大型语言模型解读学生点击流以揭示学习策略洞察 

**Authors**: Bahar Radmehr, Ekaterina Shved, Fatma Betül Güreş, Adish Singla, Tanja Käser  

**Link**: [PDF](https://arxiv.org/pdf/2505.15410)  

**Abstract**: Clickstream data from digital learning environments offer valuable insights into students' learning behaviors, but are challenging to interpret due to their high dimensionality and granularity. Prior approaches have relied mainly on handcrafted features, expert labeling, clustering, or supervised models, therefore often lacking generalizability and scalability. In this work, we introduce ClickSight, an in-context Large Language Model (LLM)-based pipeline that interprets student clickstreams to reveal their learning strategies. ClickSight takes raw clickstreams and a list of learning strategies as input and generates textual interpretations of students' behaviors during interaction. We evaluate four different prompting strategies and investigate the impact of self-refinement on interpretation quality. Our evaluation spans two open-ended learning environments and uses a rubric-based domain-expert evaluation. Results show that while LLMs can reasonably interpret learning strategies from clickstreams, interpretation quality varies by prompting strategy, and self-refinement offers limited improvement. ClickSight demonstrates the potential of LLMs to generate theory-driven insights from educational interaction data. 

**Abstract (ZH)**: 数字学习环境中的点击流数据提供了 valuable insights into 学生的学习行为，但由于其高维度和细粒度特性，解读起来具有挑战性。先前的方法主要依赖手工特征、专家标注、聚类或监督模型，因此往往缺乏通用性和可扩展性。在本文中，我们引入了 ClickSight，这是一个基于上下文的大语言模型 (LLM)-驱动的数据处理管道，用于解释学生点击流以揭示其学习策略。ClickSight 以原始点击流数据和学习策略列表作为输入，并生成学生交互过程中文本形式的解释。我们评估了四种不同的提示策略，并探究了自我润色对解释质量的影响。评估跨越了两个开放式学习环境，并使用基于评分标准的领域专家评估方法。结果表明，虽然大语言模型可以合理地从点击流数据中解释学习策略，但解释质量受提示策略的影响，并且自我润色仅提供有限的改进。ClickSight 展示了大语言模型从教育交互数据中生成理论驱动见解的潜力。 

---
# When to Continue Thinking: Adaptive Thinking Mode Switching for Efficient Reasoning 

**Title (ZH)**: 何时继续思考：高效的推理中自适应思考模式切换 

**Authors**: Xiaoyun Zhang, Jingqing Ruan, Xing Ma, Yawen Zhu, Haodong Zhao, Hao Li, Jiansong Chen, Ke Zeng, Xunliang Cai  

**Link**: [PDF](https://arxiv.org/pdf/2505.15400)  

**Abstract**: Large reasoning models (LRMs) achieve remarkable performance via long reasoning chains, but often incur excessive computational overhead due to redundant reasoning, especially on simple tasks. In this work, we systematically quantify the upper bounds of LRMs under both Long-Thinking and No-Thinking modes, and uncover the phenomenon of "Internal Self-Recovery Mechanism" where models implicitly supplement reasoning during answer generation. Building on this insight, we propose Adaptive Self-Recovery Reasoning (ASRR), a framework that suppresses unnecessary reasoning and enables implicit recovery. By introducing accuracy-aware length reward regulation, ASRR adaptively allocates reasoning effort according to problem difficulty, achieving high efficiency with negligible performance sacrifice. Experiments across multiple benchmarks and models show that, compared with GRPO, ASRR reduces reasoning budget by up to 32.5% (1.5B) and 25.7% (7B) with minimal accuracy loss (1.2% and 0.6% pass@1), and significantly boosts harmless rates on safety benchmarks (up to +21.7%). Our results highlight the potential of ASRR for enabling efficient, adaptive, and safer reasoning in LRMs. 

**Abstract (ZH)**: 大型推理模型（LRMs）通过长推理链实现显著性能，但由于冗余推理往往导致过高的计算开销，特别是在简单任务上。在本工作中，我们系统地量化了LRMs在长思考和无思考模式下的上界，并揭示了“内部自我恢复机制”的现象，即模型在答案生成过程中隐式补充推理。基于这一见解，我们提出了自适应自我恢复推理（ASRR）框架，该框架抑制不必要的推理并允许隐式恢复。通过引入基于准确性的长度奖励调节，ASRR根据问题难度自适应分配推理努力，实现了高效率且几乎无性能损失。在多个基准和模型上的实验显示，与GRPO相比，ASRR在最小化准确率损失（1.2%和0.6% pass@1）的情况下，分别将推理预算降低了32.5%（1.5B）和25.7%（7B），并在安全性基准上显著提升了无害率（最多+21.7%）。我们的结果突显了ASRR在使LRMs高效、自适应和更安全推理方面的潜力。 

---
# When Can Large Reasoning Models Save Thinking? Mechanistic Analysis of Behavioral Divergence in Reasoning 

**Title (ZH)**: 当大型推理模型能够节省思考吗？推理行为差异的机制分析 

**Authors**: Rongzhi Zhu, Yi Liu, Zequn Sun, Yiwei Wang, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15276)  

**Abstract**: Large reasoning models (LRMs) have significantly advanced performance on complex tasks, yet their tendency to overthink introduces inefficiencies. This study investigates the internal mechanisms of reinforcement learning (RL)-trained LRMs when prompted to save thinking, revealing three distinct thinking modes: no thinking (NT), explicit thinking (ET), and implicit thinking (IT). Through comprehensive analysis of confidence in thinking termination, attention from thinking to generation, and attentional focus on input sections, we uncover key factors influencing the reasoning behaviors. We further find that NT reduces output length at the cost of accuracy, while ET and IT maintain accuracy with reduced response length. Our findings expose fundamental inconsistencies in RL-optimized LRMs, necessitating adaptive improvements for reliable efficiency. 

**Abstract (ZH)**: 大型推理模型（LRMs）在复杂任务上的性能显著提升，但其过度推理的倾向引入了效率问题。本研究探讨了强化学习（RL）训练的LRMs在被提示节省推理时的内部机制，揭示了三种不同的推理模式：无推理（NT）、显式推理（ET）和隐式推理（IT）。通过对推理终止的信心、从推理到生成的关注以及输入部分的注意力焦点进行全面分析，我们发现了影响推理行为的关键因素。进一步研究表明，NT以牺牲准确性为代价减少了输出长度，而ET和IT则以减少响应长度为代价维持了准确性。我们的发现揭示了RL优化的LRMs中存在的基本不一致性，需要进行适应性改进以确保可靠的效率。 

---
# Generalised Probabilistic Modelling and Improved Uncertainty Estimation in Comparative LLM-as-a-judge 

**Title (ZH)**: 广义概率建模与比较LLM-as-judge中不确定性估计的改善 

**Authors**: Yassir Fathullah, Mark J. F. Gales  

**Link**: [PDF](https://arxiv.org/pdf/2505.15240)  

**Abstract**: This paper explores generalised probabilistic modelling and uncertainty estimation in comparative LLM-as-a-judge frameworks. We show that existing Product-of-Experts methods are specific cases of a broader framework, enabling diverse modelling options. Furthermore, we propose improved uncertainty estimates for individual comparisons, enabling more efficient selection and achieving strong performance with fewer evaluations. We also introduce a method for estimating overall ranking uncertainty. Finally, we demonstrate that combining absolute and comparative scoring improves performance. Experiments show that the specific expert model has a limited impact on final rankings but our proposed uncertainty estimates, especially the probability of reordering, significantly improve the efficiency of systems reducing the number of needed comparisons by ~50%. Furthermore, ranking-level uncertainty metrics can be used to identify low-performing predictions, where the nature of the probabilistic model has a notable impact on the quality of the overall uncertainty. 

**Abstract (ZH)**: 本文探讨了比较LLM-as-a-judge框架中广义概率建模和不确定性估计。我们展示了现有的Product-of-Experts方法是更广泛框架的特殊情形，使其具备了多样化的建模选项。此外，我们提出了一种改进的个体比较不确定性估计算法，使得系统在较少的评估下也能获得较强性能。我们还引入了一种整体排名不确定性估计方法。最后，我们证明了结合绝对评分和比较评分可以提升性能。实验显示，具体的专家模型对最终排名的影响有限，而我们提出的不确定性估计，特别是排序重排概率，显著提升了系统的效率，减少了约50%的比较次数。此外，排名级别的不确定性度量可用于识别表现不佳的预测，而概率模型的性质对整体不确定性质量有显著影响。 

---
# lmgame-Bench: How Good are LLMs at Playing Games? 

**Title (ZH)**: lmgame-Bench: 语言模型在玩游戏方面表现如何？ 

**Authors**: Lanxiang Hu, Mingjia Huo, Yuxuan Zhang, Haoyang Yu, Eric P. Xing, Ion Stoica, Tajana Rosing, Haojian Jin, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15146)  

**Abstract**: Playing video games requires perception, memory, and planning, exactly the faculties modern large language model (LLM) agents are expected to master. We study the major challenges in using popular video games to evaluate modern LLMs and find that directly dropping LLMs into games cannot make an effective evaluation, for three reasons -- brittle vision perception, prompt sensitivity, and potential data contamination. We introduce lmgame-Bench to turn games into reliable evaluations. lmgame-Bench features a suite of platformer, puzzle, and narrative games delivered through a unified Gym-style API and paired with lightweight perception and memory scaffolds, and is designed to stabilize prompt variance and remove contamination. Across 13 leading models, we show lmgame-Bench is challenging while still separating models well. Correlation analysis shows that every game probes a unique blend of capabilities often tested in isolation elsewhere. More interestingly, performing reinforcement learning on a single game from lmgame-Bench transfers both to unseen games and to external planning tasks. Our evaluation code is available at this https URL. 

**Abstract (ZH)**: 使用流行视频游戏评估现代大语言模型面临的主要挑战及lmgame-Bench解决方案 

---
# ModelingAgent: Bridging LLMs and Mathematical Modeling for Real-World Challenges 

**Title (ZH)**: ModelingAgent: 联接大规模语言模型与数学建模以应对现实世界挑战 

**Authors**: Cheng Qian, Hongyi Du, Hongru Wang, Xiusi Chen, Yuji Zhang, Avirup Sil, Chengxiang Zhai, Kathleen McKeown, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.15068)  

**Abstract**: Recent progress in large language models (LLMs) has enabled substantial advances in solving mathematical problems. However, existing benchmarks often fail to reflect the complexity of real-world problems, which demand open-ended, interdisciplinary reasoning and integration of computational tools. To address this gap, we introduce ModelingBench, a novel benchmark featuring real-world-inspired, open-ended problems from math modeling competitions across diverse domains, ranging from urban traffic optimization to ecosystem resource planning. These tasks require translating natural language into formal mathematical formulations, applying appropriate tools, and producing structured, defensible reports. ModelingBench also supports multiple valid solutions, capturing the ambiguity and creativity of practical modeling. We also present ModelingAgent, a multi-agent framework that coordinates tool use, supports structured workflows, and enables iterative self-refinement to generate well-grounded, creative solutions. To evaluate outputs, we further propose ModelingJudge, an expert-in-the-loop system leveraging LLMs as domain-specialized judges assessing solutions from multiple expert perspectives. Empirical results show that ModelingAgent substantially outperforms strong baselines and often produces solutions indistinguishable from those of human experts. Together, our work provides a comprehensive framework for evaluating and advancing real-world problem-solving in open-ended, interdisciplinary modeling challenges. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）的进步促进了数学问题求解的显著进展。然而，现有的基准往往未能反映现实世界问题的复杂性，这些现实世界的问题需要开放式的、跨学科的推理，并且需要集成计算工具。为解决这一差距，我们引入了ModelingBench，这是一个新颖的基准，包含来自不同领域数学建模竞赛的真实世界启发式、开放性问题，从城市交通优化到生态系统资源规划不等。这些任务要求将自然语言转化为正式的数学公式，应用适当的工具，并生成结构化的、有说服力的报告。ModelingBench还支持多种有效解法，捕捉实际建模中的模糊性和创造性。我们还提出了ModelingAgent，这是一种多agent框架，协调工具的使用，支持结构化的流程，并启用迭代自我完善以生成坚实而有创意的解决方案。为了评估输出，我们进一步提出了ModelingJudge，这是一种循环专家系统的概念，利用LLM作为领域专门化的裁判，从多个专家的角度评估解决方案。实验证明，ModelingAgent显著优于强基线，并且往往生成与人类专家相当的解决方案。我们的工作提供了一个全面的框架，用于评估和促进开放式的、跨学科的建模挑战中的实际问题解决。 

---
# Self-Evolving Curriculum for LLM Reasoning 

**Title (ZH)**: 自适应演化课程体系for大规模语言模型推理 

**Authors**: Xiaoyin Chen, Jiarui Lu, Minsu Kim, Dinghuai Zhang, Jian Tang, Alexandre Piché, Nicolas Gontier, Yoshua Bengio, Ehsan Kamalloo  

**Link**: [PDF](https://arxiv.org/pdf/2505.14970)  

**Abstract**: Reinforcement learning (RL) has proven effective for fine-tuning large language models (LLMs), significantly enhancing their reasoning abilities in domains such as mathematics and code generation. A crucial factor influencing RL fine-tuning success is the training curriculum: the order in which training problems are presented. While random curricula serve as common baselines, they remain suboptimal; manually designed curricula often rely heavily on heuristics, and online filtering methods can be computationally prohibitive. To address these limitations, we propose Self-Evolving Curriculum (SEC), an automatic curriculum learning method that learns a curriculum policy concurrently with the RL fine-tuning process. Our approach formulates curriculum selection as a non-stationary Multi-Armed Bandit problem, treating each problem category (e.g., difficulty level or problem type) as an individual arm. We leverage the absolute advantage from policy gradient methods as a proxy measure for immediate learning gain. At each training step, the curriculum policy selects categories to maximize this reward signal and is updated using the TD(0) method. Across three distinct reasoning domains: planning, inductive reasoning, and mathematics, our experiments demonstrate that SEC significantly improves models' reasoning capabilities, enabling better generalization to harder, out-of-distribution test problems. Additionally, our approach achieves better skill balance when fine-tuning simultaneously on multiple reasoning domains. These findings highlight SEC as a promising strategy for RL fine-tuning of LLMs. 

**Abstract (ZH)**: 强化学习（RL）已被证明有效于微调大规模语言模型（LLMs），显著增强了其在数学和代码生成等领域的推理能力。影响RL微调成功的关键因素是训练课程：训练问题的呈现顺序。虽然随机课程作为常见的基线方法，但仍然不尽完美；手动设计的课程往往依赖于启发式方法，而在线过滤方法可能计算成本高昂。为解决这些问题，我们提出了一种自动课程学习方法——自我演化课程（SEC），该方法在RL微调过程中并行学习课程策略。我们的方法将课程选择形式化为非平稳多臂 bandit 问题，将每个问题类别（例如，难度级别或问题类型）视为一个独立的臂。我们利用策略梯度方法的绝对优势作为即时学习增益的代理度量。在每一步训练中，课程策略选择最大化此奖励信号的类别，并使用TD(0)方法进行更新。在推理领域（规划、归纳推理和数学）的三个不同领域中，我们的实验表明，SEC 显著提高了模型的推理能力，使其在更难的、分布外的测试问题上表现出更好的泛化能力。此外，当我们同时在多个推理领域进行微调时，我们的方法在技能平衡方面表现更佳。这些发现突显了SEC作为LLMs的RL微调的一种有前途的策略。 

---
# Reinforcement Learning from User Feedback 

**Title (ZH)**: 用户反馈驱动的强化学习 

**Authors**: Eric Han, Jun Chen, Karthik Abinav Sankararaman, Xiaoliang Peng, Tengyu Xu, Eryk Helenowski, Kaiyan Peng, Mrinal Kumar, Sinong Wang, Han Fang, Arya Talebzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2505.14946)  

**Abstract**: As large language models (LLMs) are increasingly deployed in diverse user facing applications, aligning them with real user preferences becomes essential. Existing methods like Reinforcement Learning from Human Feedback (RLHF) rely on expert annotators trained on manually defined guidelines, whose judgments may not reflect the priorities of everyday users. We introduce Reinforcement Learning from User Feedback (RLUF), a framework for aligning LLMs directly to implicit signals from users in production. RLUF addresses key challenges of user feedback: user feedback is often binary (e.g., emoji reactions), sparse, and occasionally adversarial. We train a reward model, P[Love], to predict the likelihood that an LLM response will receive a Love Reaction, a lightweight form of positive user feedback, and integrate P[Love] into a multi-objective policy optimization framework alongside helpfulness and safety objectives. In large-scale experiments, we show that P[Love] is predictive of increased positive feedback and serves as a reliable offline evaluator of future user behavior. Policy optimization using P[Love] significantly raises observed positive-feedback rates, including a 28% increase in Love Reactions during live A/B tests. However, optimizing for positive reactions introduces reward hacking challenges, requiring careful balancing of objectives. By directly leveraging implicit signals from users, RLUF offers a path to aligning LLMs with real-world user preferences at scale. 

**Abstract (ZH)**: 基于用户反馈的强化学习（RLUF）：一种直接将大型语言模型与生产中的隐式用户信号对齐的框架 

---
# FOL-Pretrain: A complexity annotated corpus of first-order logic 

**Title (ZH)**: FOL-预训练：带有复杂性标注的一阶逻辑语料库 

**Authors**: Isabelle Lee, Sarah Liaw, Dani Yogatama  

**Link**: [PDF](https://arxiv.org/pdf/2505.14932)  

**Abstract**: Transformer-based large language models (LLMs) have demonstrated remarkable reasoning capabilities such as coding and solving mathematical problems to commonsense inference. While these tasks vary in complexity, they all require models to integrate and compute over structured information. Despite recent efforts to reverse-engineer LLM behavior through controlled experiments, our understanding of how these models internalize and execute complex algorithms remains limited. Progress has largely been confined to small-scale studies or shallow tasks such as basic arithmetic and grammatical pattern matching. One barrier to deeper understanding is the nature of pretraining data -- vast, heterogeneous, and often poorly annotated, making it difficult to isolate mechanisms of reasoning. To bridge this gap, we introduce a large-scale, fully open, complexity-annotated dataset of first-order logic reasoning traces, designed to probe and analyze algorithmic reasoning in LLMs. The dataset consists of 3.5 billion tokens, including 8.8 million LLM-augmented, human-annotated examples and 7.5 million synthetically generated examples. Each synthetic example is verifiably correct, produced by a custom automated theorem solver, and accompanied by metadata tracing its algorithmic provenance. We aim to provide a scalable, interpretable artifact for studying how LLMs learn and generalize symbolic reasoning processes, paving the way for more transparent and targeted investigations into the algorithmic capabilities of modern models. 

**Abstract (ZH)**: 基于Transformer的大语言模型（LLM）展示了 remarkable 的推理能力，包括编程、解决数学问题到常识推理。尽管这些任务在复杂性上有所不同，但都需要模型整合和计算结构化信息。尽管最近通过受控实验反向工程LLM的行为取得了进步，但我们对这些模型如何 internalize 和执行复杂算法的理解仍然有限。进展主要局限于小型研究或简单的任务，如基本算术和语法模式匹配。一个深入了解的障碍是预训练数据的性质——庞大、异构且通常标注不佳，这使得难以隔离推理机制。为弥补这一差距，我们引入了一个大规模、全开放、复杂性标注的一阶逻辑推理跟踪数据集，旨在探究和分析LLM的算法推理。该数据集包含35亿个标记，包括880万个人工标注的LLM增强示例和7500万个合成生成的示例。每个合成示例都由自定义自动定理求解器验证正确，并附有追踪其算法起源的元数据。我们旨在提供一个可扩展、可解释的工具，用于研究LLM如何学习和泛化符号推理过程，为更透明和有针对性地探究现代模型的算法能力铺平道路。 

---
# R&D-Agent: Automating Data-Driven AI Solution Building Through LLM-Powered Automated Research, Development, and Evolution 

**Title (ZH)**: R&D-Agent: 通过LLM赋能的自动化数据驱动AI解决方案构建、开发与进化 

**Authors**: Xu Yang, Xiao Yang, Shikai Fang, Bowen Xian, Yuante Li, Jian Wang, Minrui Xu, Haoran Pan, Xinpeng Hong, Weiqing Liu, Yelong Shen, Weizhu Chen, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2505.14738)  

**Abstract**: Recent advances in AI and ML have transformed data science, yet increasing complexity and expertise requirements continue to hinder progress. While crowdsourcing platforms alleviate some challenges, high-level data science tasks remain labor-intensive and iterative. To overcome these limitations, we introduce R&D-Agent, a dual-agent framework for iterative exploration. The Researcher agent uses performance feedback to generate ideas, while the Developer agent refines code based on error feedback. By enabling multiple parallel exploration traces that merge and enhance one another, R&D-Agent narrows the gap between automated solutions and expert-level performance. Evaluated on MLE-Bench, R&D-Agent emerges as the top-performing machine learning engineering agent, demonstrating its potential to accelerate innovation and improve precision across diverse data science applications. We have open-sourced R&D-Agent on GitHub: this https URL. 

**Abstract (ZH)**: 最近在人工智能和机器学习方面的进展已转变了数据科学，但不断增加的复杂性和专业知识要求依然阻碍着进步。尽管众包平台可以缓解一些挑战，但高级数据科学任务依然劳动密集且需要迭代。为克服这些限制，我们引入了R&D-Agent，这是一种用于迭代探索的双代理框架。研究员代理利用性能反馈生成想法，而开发者代理基于错误反馈改进代码。通过启用多个并行的探索轨迹并相互融合和增强，R&D-Agent 缩小了自动化解决方案与专家级性能之间的差距。在MLE-Bench上的评估表明，R&D-Agent 是表现最佳的机器学习工程代理，展示了其在促进创新并提高各种数据科学应用精度方面的潜力。我们已在GitHub上开源了R&D-Agent：this https URL。 

---
# VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models 

**Title (ZH)**: VerifyBench：基于引用奖励系统的大型语言模型基准测试 

**Authors**: Yuchen Yan, Jin Jiang, Zhenbang Ren, Yijun Li, Xudong Cai, Yang Liu, Xin Xu, Mengdi Zhang, Jian Shao, Yongliang Shen, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15801)  

**Abstract**: Large reasoning models such as OpenAI o1 and DeepSeek-R1 have achieved remarkable performance in the domain of reasoning. A key component of their training is the incorporation of verifiable rewards within reinforcement learning (RL). However, existing reward benchmarks do not evaluate reference-based reward systems, leaving researchers with limited understanding of the accuracy of verifiers used in RL. In this paper, we introduce two benchmarks, VerifyBench and VerifyBench-Hard, designed to assess the performance of reference-based reward systems. These benchmarks are constructed through meticulous data collection and curation, followed by careful human annotation to ensure high quality. Current models still show considerable room for improvement on both VerifyBench and VerifyBench-Hard, especially smaller-scale models. Furthermore, we conduct a thorough and comprehensive analysis of evaluation results, offering insights for understanding and developing reference-based reward systems. Our proposed benchmarks serve as effective tools for guiding the development of verifier accuracy and the reasoning capabilities of models trained via RL in reasoning tasks. 

**Abstract (ZH)**: Large Reasoning Models Such as OpenAI o1 and DeepSeek-R1 Have Achieved Remarkable Performance in the Domain of Reasoning: Introducing VerifyBench and VerifyBench-Hard to Assess Reference-Based Reward Systems 

---
# Large Language Models as Computable Approximations to Solomonoff Induction 

**Title (ZH)**: 大型语言模型作为索洛莫诺夫归纳的可计算近似 

**Authors**: Jun Wan, Lingrui Mei  

**Link**: [PDF](https://arxiv.org/pdf/2505.15784)  

**Abstract**: The rapid advancement of large language models (LLMs) calls for a rigorous theoretical framework to explain their empirical success. While significant progress has been made in understanding LLM behaviors, existing theoretical frameworks remain fragmented in explaining emergent phenomena through a unified mathematical lens. We establish the first formal connection between LLM architectures and Algorithmic Information Theory (AIT) by proving two fundamental results: (1) the training process computationally approximates Solomonoff prior through loss minimization interpreted as program length optimization, and (2) next-token prediction implements approximate Solomonoff induction. We leverage AIT to provide a unified theoretical explanation for in-context learning, few-shot learning, and scaling laws. Furthermore, our theoretical insights lead to a principled method for few-shot example selection that prioritizes samples where models exhibit lower predictive confidence. We demonstrate through experiments on diverse text classification benchmarks that this strategy yields significant performance improvements, particularly for smaller model architectures, when compared to selecting high-confidence examples. Our framework bridges the gap between theoretical foundations and practical LLM behaviors, providing both explanatory power and actionable insights for future model development. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅速发展呼唤一个严谨的理论框架来解释其 empirical 成功。虽然在理解 LLM 行为方面已经取得显著进展，但现有的理论框架仍缺乏通过统一的数学视角来解释新兴现象的能力。我们通过证明两个基本结果，首次正式建立了 LLM 架构与算法信息论（AIT）之间的联系：（1）训练过程通过对损失最小化进行计算性近似，将索罗门off 先验近似为程序长度优化，（2）下一个 token 预测实现了近似的索罗门off 归纳。我们利用AIT提供了一个统一的理论解释，包括上下文学习、少样本学习和标度规律。此外，我们的理论见解导致了一种原则性的少样本示例选择方法，优先选择模型预测置信度较低的样本。通过在各种文本分类基准上的实验，我们证明了这种方法在与选择高置信度示例相比时，对于较小的模型架构尤其能够带来显著的性能提升。我们的框架在理论基础与实际 LLM 行为之间架起了桥梁，不仅提供了解释力，还提供了对未来模型开发的实际洞察。 

---
# Soft Thinking: Unlocking the Reasoning Potential of LLMs in Continuous Concept Space 

**Title (ZH)**: 软思考：在连续概念空间中解锁LLMs的推理潜力 

**Authors**: Zhen Zhang, Xuehai He, Weixiang Yan, Ao Shen, Chenyang Zhao, Shuohang Wang, Yelong Shen, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15778)  

**Abstract**: Human cognition typically involves thinking through abstract, fluid concepts rather than strictly using discrete linguistic tokens. Current reasoning models, however, are constrained to reasoning within the boundaries of human language, processing discrete token embeddings that represent fixed points in the semantic space. This discrete constraint restricts the expressive power and upper potential of such reasoning models, often causing incomplete exploration of reasoning paths, as standard Chain-of-Thought (CoT) methods rely on sampling one token per step. In this work, we introduce Soft Thinking, a training-free method that emulates human-like "soft" reasoning by generating soft, abstract concept tokens in a continuous concept space. These concept tokens are created by the probability-weighted mixture of token embeddings, which form the continuous concept space, enabling smooth transitions and richer representations that transcend traditional discrete boundaries. In essence, each generated concept token encapsulates multiple meanings from related discrete tokens, implicitly exploring various reasoning paths to converge effectively toward the correct answer. Empirical evaluations on diverse mathematical and coding benchmarks consistently demonstrate the effectiveness and efficiency of Soft Thinking, improving pass@1 accuracy by up to 2.48 points while simultaneously reducing token usage by up to 22.4% compared to standard CoT. Qualitative analysis further reveals that Soft Thinking outputs remain highly interpretable and readable, highlighting the potential of Soft Thinking to break the inherent bottleneck of discrete language-based reasoning. Code is available at this https URL. 

**Abstract (ZH)**: 人类认知通常涉及通过抽象且流动的概念进行思考，而不是严格地使用离散的语言标记。然而，当前的推理模型仅限于在人类语言的框架内进行推理，处理表示语义空间中固定点的离散标记嵌入。这种离散约束限制了此类推理模型的表达能力和潜在能力，常常导致推理路径探索不完整，因为标准思维链（CoT）方法依赖于每步采样一个标记。在本文中，我们提出了软思考（Soft Thinking）这一无需训练的方法，通过在连续的概念空间中生成软的抽象概念标记来模拟人类的“软”推理。这些概念标记是由标记嵌入的概率加权混合生成的，形成了连续的概念空间，这使得可以实现平滑过渡和更丰富的表示，超越了传统离散边界的限制。本质上而言，每个生成的概念标记封装了相关离散标记的多种含义，隐含地探索各种推理路径以有效收敛于正确答案。在不同的数学和编码基准测试上的实证评估一致表明，软思考的有效性和效率，与标准CoT相比，软思考可以提高pass@1准确率高达2.48个百分点，同时降低标记使用率高达22.4%。定性分析进一步表明，软思考的输出保持高度可解释性和可读性，突显了软思考在打破基于离散语言推理的固有问题方面的潜力。代码可在此处访问：this https URL。 

---
# Scalable Defense against In-the-wild Jailbreaking Attacks with Safety Context Retrieval 

**Title (ZH)**: 基于安全上下文检索的大规模防御在野*jailbreaking*攻击 

**Authors**: Taiye Chen, Zeming Wei, Ang Li, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15753)  

**Abstract**: Large Language Models (LLMs) are known to be vulnerable to jailbreaking attacks, wherein adversaries exploit carefully engineered prompts to induce harmful or unethical responses. Such threats have raised critical concerns about the safety and reliability of LLMs in real-world deployment. While existing defense mechanisms partially mitigate such risks, subsequent advancements in adversarial techniques have enabled novel jailbreaking methods to circumvent these protections, exposing the limitations of static defense frameworks. In this work, we explore defending against evolving jailbreaking threats through the lens of context retrieval. First, we conduct a preliminary study demonstrating that even a minimal set of safety-aligned examples against a particular jailbreak can significantly enhance robustness against this attack pattern. Building on this insight, we further leverage the retrieval-augmented generation (RAG) techniques and propose Safety Context Retrieval (SCR), a scalable and robust safeguarding paradigm for LLMs against jailbreaking. Our comprehensive experiments demonstrate how SCR achieves superior defensive performance against both established and emerging jailbreaking tactics, contributing a new paradigm to LLM safety. Our code will be available upon publication. 

**Abstract (ZH)**: 大型语言模型（LLMs）易受 Jailbreaking 攻击，攻击者通过精心设计的提示诱导有害或不道德的响应。这类威胁引发了对 LLMs 在实际部署中的安全性和可靠性的严重担忧。尽管现有防御机制部分缓解了此类风险，但随后对手技术的进展使得新的 Jailbreaking 方法能够绕过这些防护，暴露出静态防御框架的局限性。在本工作中，我们从上下文检索的角度探索防范 evolving jailbreaking 威胁的方法。首先，我们进行初步研究，表明针对某特定 Jailbreak 的少量安全对齐示例可以显著增强对该攻击模式的鲁棒性。基于这一洞见，我们进一步利用检索增强生成（RAG）技术，并提出安全上下文检索（SCR），这是一种可扩展且稳健的 LLM 防护范式，以抵御 Jailbreaking。我们的全面实验展示了 SCR 如何在对抗既有的和新兴的 Jailbreaking 技巧时实现更优的防御性能，为 LLM 安全性贡献了一个新范式。我们的代码将在发表后提供。 

---
# HybridProver: Augmenting Theorem Proving with LLM-Driven Proof Synthesis and Refinement 

**Title (ZH)**: HybridProver: 结合LLM驱动的证明合成与 refinement 的定理证明方法 

**Authors**: Jilin Hu, Jianyu Zhang, Yongwang Zhao, Talia Ringer  

**Link**: [PDF](https://arxiv.org/pdf/2505.15740)  

**Abstract**: Formal methods is pivotal for verifying the reliability of critical systems through rigorous mathematical proofs. However, its adoption is hindered by labor-intensive manual proofs and the expertise required to use theorem provers. Recent advancements in large language models (LLMs) offer new opportunities for automated theorem proving. Two promising approaches are generating tactics step by step and generating a whole proof directly with an LLM. However, existing work makes no attempt to combine the two approaches. In this work, we introduce HybridProver, a dual-model proof synthesis framework that combines tactic-based generation and whole-proof synthesis to harness the benefits of both approaches. HybridProver generates whole proof candidates for evaluation directly, then extracts proof sketches from those candidates. It then uses a tactic-based generation model that integrates automated tools to complete the sketches via stepwise refinement. We implement HybridProver for the Isabelle theorem prover and fine-tune LLMs on our optimized Isabelle datasets. Evaluation on the miniF2F dataset illustrates HybridProver's effectiveness. We achieve a 59.4% success rate on miniF2F, where the previous SOTA is 56.1%. Our ablation studies show that this SOTA result is attributable to combining whole-proof and tactic-based generation. Additionally, we show how the dataset quality, training parameters, and sampling diversity affect the final result during automated theorem proving with LLMs. All of our code, datasets, and LLMs are open source. 

**Abstract (ZH)**: 形式化方法对于通过严格的数学证明验证关键系统可靠性至关重要。然而，其采用受到劳动密集型的手动证明和使用定理证明器所需的专门知识的阻碍。近期大型语言模型的进展为自动定理证明提供了新机会。两种有前景的方法是通过逐步生成策略和直接使用大型语言模型生成完整证明。然而，现有研究并未尝试将这两种方法结合起来。在本文中，我们提出了HybridProver，这是一种结合基于策略的生成和整体证明合成的双模型证明合成框架，以利用这两种方法的优势。HybridProver直接生成用于评估的整体证明候选，然后从中提取证明草图。接着使用结合自动工具的基于策略的生成模型，通过逐步细化完成这些草图。我们为Isabelle定理证明器实现HybridProver，并在我们优化的Isabelle数据集上微调大型语言模型。对miniF2F数据集的评估展示了HybridProver的有效性。我们在miniF2F上实现了59.4%的成功率，而之前的最佳成果（SOTA）是56.1%。我们的消融研究显示，这一最佳成果归因于结合整体证明和基于策略的生成。此外，我们展示了数据集质量、训练参数和采样多样性如何影响使用大型语言模型进行自动定理证明的最终结果。所有我们的代码、数据集和大型语言模型都是开源的。 

---
# Alignment Under Pressure: The Case for Informed Adversaries When Evaluating LLM Defenses 

**Title (ZH)**: 在压力之下对齐：评估LLM防御措施时需要有知识的对手的理由 

**Authors**: Xiaoxue Yang, Bozhidar Stevanoski, Matthieu Meeus, Yves-Alexandre de Montjoye  

**Link**: [PDF](https://arxiv.org/pdf/2505.15738)  

**Abstract**: Large language models (LLMs) are rapidly deployed in real-world applications ranging from chatbots to agentic systems. Alignment is one of the main approaches used to defend against attacks such as prompt injection and jailbreaks. Recent defenses report near-zero Attack Success Rates (ASR) even against Greedy Coordinate Gradient (GCG), a white-box attack that generates adversarial suffixes to induce attacker-desired outputs. However, this search space over discrete tokens is extremely large, making the task of finding successful attacks difficult. GCG has, for instance, been shown to converge to local minima, making it sensitive to initialization choices. In this paper, we assess the future-proof robustness of these defenses using a more informed threat model: attackers who have access to some information about the alignment process. Specifically, we propose an informed white-box attack leveraging the intermediate model checkpoints to initialize GCG, with each checkpoint acting as a stepping stone for the next one. We show this approach to be highly effective across state-of-the-art (SOTA) defenses and models. We further show our informed initialization to outperform other initialization methods and show a gradient-informed checkpoint selection strategy to greatly improve attack performance and efficiency. Importantly, we also show our method to successfully find universal adversarial suffixes -- single suffixes effective across diverse inputs. Our results show that, contrary to previous beliefs, effective adversarial suffixes do exist against SOTA alignment-based defenses, that these can be found by existing attack methods when adversaries exploit alignment knowledge, and that even universal suffixes exist. Taken together, our results highlight the brittleness of current alignment-based methods and the need to consider stronger threat models when testing the safety of LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在从聊天bot到代理系统等多种现实应用中迅速部署。对齐是防御诸如提示注入和 Jailbreak 等攻击的主要方法之一。最近的研究报告即使在 Greedy Coordinate Gradient（GCG）这种白盒攻击（它可以生成对抗后缀以诱导攻击者期望的输出）面前，防御措施也能实现几乎零的攻击成功率（ASR）。然而，随着对离散令牌搜索空间的扩大，成功攻击的发现任务变得极具挑战性。GCG 例如已被证明会收敛到局部极小值，使其对初始化的选择高度敏感。在本文中，我们使用更明智的威胁模型来评估这些防御措施的未来稳健性：拥有对对齐过程某些信息访问权限的攻击者。具体地，我们提出了一个利用中间模型检查点进行初始化的有信息的白盒攻击方案，每个检查点作为下一个检查点的踏脚石。我们表明该方法对最先进的（SOTA）防御措施和模型具有高度有效性。进一步显示我们的有信息初始化方法优于其他初始化方法，并提出了基于梯度的检查点选择策略，以大幅提高攻击性能和效率。重要的是，我们还展示了我们的方法能够找到通用的对抗后缀——这些后缀在多种输入下均可生效。我们的结果表明，与以往认为不同，有效的对抗后缀确实存在于对抗 SOTA 对齐基线防御措施的情况中；这些后缀在对手利用对齐知识时能够被现有攻击方法找到；甚至通用的后缀也确实存在。综合来看，我们的结果揭示了当前对齐基线方法的脆弱性，并强调在测试 LLM 安全性时需要考虑更强的威胁模型的重要性。 

---
# DEBATE, TRAIN, EVOLVE: Self Evolution of Language Model Reasoning 

**Title (ZH)**: 辩论、训练、演变：语言模型的自我进化 

**Authors**: Gaurav Srivastava, Zhenyu Bi, Meng Lu, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15734)  

**Abstract**: Large language models (LLMs) have improved significantly in their reasoning through extensive training on massive datasets. However, relying solely on additional data for improvement is becoming increasingly impractical, highlighting the need for models to autonomously enhance their reasoning without external supervision. In this paper, we propose Debate, Train, Evolve (DTE), a novel ground truth-free training framework that uses multi-agent debate traces to evolve a single language model. We also introduce a new prompting strategy Reflect-Critique-Refine, to improve debate quality by explicitly instructing agents to critique and refine their reasoning. Extensive evaluations on five reasoning benchmarks with six open-weight models show that our DTE framework achieve substantial improvements, with an average accuracy gain of 8.92% on the challenging GSM-PLUS dataset. Furthermore, we observe strong cross-domain generalization, with an average accuracy gain of 5.8% on all other benchmarks, suggesting that our method captures general reasoning capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过大量数据的广泛训练，在推理方面取得了显著改进。然而，仅依赖额外数据进行改进正变得越来越不现实，凸显了模型在无需外部监督的情况下自主提升推理能力的需求。本文提出了一种名为Debate, Train, Evolve (DTE)的新颖无地面真值训练框架，该框架使用多代理辩论轨迹来进化单一语言模型。我们还引入了一种新的提示策略Reflect-Critique-Refine，通过明确指示代理批判和改进推理来提高辩论质量。在五个推理基准上的广泛评估显示，我们的DTE框架取得了显著改进，在具有挑战性的GSM-PLUS数据集上平均准确率提高了8.92%。此外，我们观察到强烈的跨领域泛化能力，在其他所有基准上平均准确率提高了5.8%，表明我们的方法捕捉到了通用的推理能力。 

---
# Shared Path: Unraveling Memorization in Multilingual LLMs through Language Similarities 

**Title (ZH)**: 共享路径：通过语言相似性揭开多语言LLM中的记忆机制 

**Authors**: Xiaoyu Luo, Yiyi Chen, Johannes Bjerva, Qiongxiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.15722)  

**Abstract**: We present the first comprehensive study of Memorization in Multilingual Large Language Models (MLLMs), analyzing 95 languages using models across diverse model scales, architectures, and memorization definitions. As MLLMs are increasingly deployed, understanding their memorization behavior has become critical. Yet prior work has focused primarily on monolingual models, leaving multilingual memorization underexplored, despite the inherently long-tailed nature of training corpora. We find that the prevailing assumption, that memorization is highly correlated with training data availability, fails to fully explain memorization patterns in MLLMs. We hypothesize that treating languages in isolation - ignoring their similarities - obscures the true patterns of memorization. To address this, we propose a novel graph-based correlation metric that incorporates language similarity to analyze cross-lingual memorization. Our analysis reveals that among similar languages, those with fewer training tokens tend to exhibit higher memorization, a trend that only emerges when cross-lingual relationships are explicitly modeled. These findings underscore the importance of a language-aware perspective in evaluating and mitigating memorization vulnerabilities in MLLMs. This also constitutes empirical evidence that language similarity both explains Memorization in MLLMs and underpins Cross-lingual Transferability, with broad implications for multilingual NLP. 

**Abstract (ZH)**: 我们首次对多语言大型语言模型（MLLMs）中的记忆行为进行全面研究，分析了95种语言，并使用了不同规模、架构和记忆定义的各种模型。随着MLLMs的日益部署，理解其记忆行为变得至关重要。然而，以往的工作主要集中在单语言模型上，导致多语言记忆行为的研究不足，尽管训练语料库 inherently 呈长尾分布。我们发现，记忆与训练数据可用性的高度相关性这一主导假设，并不能完全解释MLLMs的记忆模式。我们假设将语言视为孤立个体，忽略它们的相似性，会掩盖真正的记忆模式。为了应对这一问题，我们提出了一种基于图的关联度量，该度量整合了语言相似性以分析跨语言记忆。我们的分析揭示，在相似语言中，训练令牌较少的语言更容易显示更高的记忆现象，只有在显式建模跨语言关系时这一趋势才得以显现。这些发现强调了在评估和缓解MLLMs的记忆漏洞时，采用语言 Awareness 观点的重要性。这也提供了实证证据，表明语言相似性不仅是MLLMs中记忆现象的解释，也是跨语言可迁移性的基石，对多语言自然语言处理有广泛影响。 

---
# A Federated Splitting Framework for LLMs: Security, Efficiency, and Adaptability 

**Title (ZH)**: 联邦分割框架for LLMs：安全、效率与适应性 

**Authors**: Zishuai Zhang, Hainan Zhang, Jiaying Zheng, Ziwei Wang, Yongxin Tong, Jin Dong, Zhiming Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.15683)  

**Abstract**: Private data is typically larger and of higher quality than public data, offering great potential to improve LLM. However, its scattered distribution across data silos and the high computational demands of LLMs limit their deployment in federated environments. To address this, the transformer-based split learning model has emerged, offloading most model parameters to the server while retaining only the embedding and output layers on clients to ensure privacy. However, it still faces significant challenges in security, efficiency, and adaptability: 1) embedding gradients are vulnerable to attacks, leading to reverse engineering of private data; 2) the autoregressive nature of LLMs means that federated split learning can only train and infer sequentially, causing high communication overhead; 3) fixed partition points lack adaptability to downstream tasks. In this paper, we introduce FL-LLaMA, a secure, efficient, and adaptive federated split framework based on LLaMA2. First, we place some input and output blocks on the local client and inject Gaussian noise into forward-pass hidden states, enabling secure end-to-end propagation. Second, we employ client-batch and server-hierarchical strategies to achieve parallel training, along with attention-mask compression and KV cache mechanisms to accelerate inference, reducing communication costs effectively. Third, we allow users to dynamically adjust the partition points for input/output blocks based on specific task requirements and hardware limitations. Experiments on NLU, summarization and conversational QA tasks show that FL-LLaMA maintains performance comparable to centralized LLaMA2, and achieves up to 2x train speedups and 8x inference speedups. Further analysis of privacy attacks and different partition points also demonstrates the effectiveness of FL-LLaMA in security and adaptability. 

**Abstract (ZH)**: 基于LLaMA2的FL-LLaMA：一种安全、高效且适应性强的联邦分学习框架 

---
# UniErase: Unlearning Token as a Universal Erasure Primitive for Language Models 

**Title (ZH)**: UniErase: 作为一种通用擦除原语的语言模型去学习 token 

**Authors**: Miao Yu, Liang Lin, Guibin Zhang, Xinfeng Li, Junfeng Fang, Ningyu Zhang, Kun Wang, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15674)  

**Abstract**: Large language models require iterative updates to address challenges such as knowledge conflicts and outdated information (e.g., incorrect, private, or illegal contents). Machine unlearning provides a systematic methodology for targeted knowledge removal from trained models, enabling elimination of sensitive information influences. However, mainstream fine-tuning-based unlearning methods often fail to balance unlearning efficacy and model ability, frequently resulting in catastrophic model collapse under extensive knowledge removal. Meanwhile, in-context unlearning, which relies solely on contextual prompting without modifying the model's intrinsic mechanisms, suffers from limited generalizability and struggles to achieve true unlearning. In this work, we introduce UniErase, a novel unlearning paradigm that employs learnable parametric suffix (unlearning token) to steer language models toward targeted forgetting behaviors. UniErase operates through two key phases: (I) an optimization stage that binds desired unlearning outputs to the model's autoregressive probability distribution via token optimization, followed by (II) a lightweight model editing phase that activates the learned token to probabilistically induce specified forgetting objective. Serving as a new research direction for token learning to induce unlearning target, UniErase achieves state-of-the-art (SOTA) performance across batch, sequential, and precise unlearning under fictitious and real-world knowledge settings. Remarkably, in terms of TOFU benchmark, UniErase, modifying only around 3.66% of the LLM parameters, outperforms previous forgetting SOTA baseline by around 4.01 times for model ability with even better unlearning efficacy. Similarly, UniErase, maintaining more ability, also surpasses previous retaining SOTA by 35.96% for unlearning efficacy, showing dual top-tier performances in current unlearing domain. 

**Abstract (ZH)**: UniErase：一种用于诱导遗忘目标的可学习参数后缀新范式 

---
# Listen to the Context: Towards Faithful Large Language Models for Retrieval Augmented Generation on Climate Questions 

**Title (ZH)**: 倾听上下文：面向气候问题检索增强生成的忠实大型语言模型 

**Authors**: David Thulke, Jakob Kemmler, Christian Dugast, Hermann Ney  

**Link**: [PDF](https://arxiv.org/pdf/2505.15633)  

**Abstract**: Large language models that use retrieval augmented generation have the potential to unlock valuable knowledge for researchers, policymakers, and the public by making long and technical climate-related documents more accessible. While this approach can help alleviate factual hallucinations by relying on retrieved passages as additional context, its effectiveness depends on whether the model's output remains faithful to these passages. To address this, we explore the automatic assessment of faithfulness of different models in this setting. We then focus on ClimateGPT, a large language model specialised in climate science, to examine which factors in its instruction fine-tuning impact the model's faithfulness. By excluding unfaithful subsets of the model's training data, we develop ClimateGPT Faithful+, which achieves an improvement in faithfulness from 30% to 57% in supported atomic claims according to our automatic metric. 

**Abstract (ZH)**: 使用检索增强生成的大语言模型有可能通过使与气候相关的长篇技术文档更易于访问，为研究人员、 Policymakers 和公众解锁有价值的knowledge。虽然这种方法可以通过依赖检索段落作为额外背景信息来缓解事实幻觉的问题，但其效果取决于模型的输出是否忠于这些段落。为了解决这个问题，我们探索了在这种情境下自动评估不同模型忠实度的方法。然后，我们专注于专门从事气候科学的大型语言模型ClimateGPT，研究其指令微调中的哪些因素影响模型的忠实度。通过排除不忠实的模型训练数据子集，我们开发了ClimateGPT Faithful+，在我们自动评估指标的支持原子声明中，其忠实度从30%提升到57%。 

---
# From Problem-Solving to Teaching Problem-Solving: Aligning LLMs with Pedagogy using Reinforcement Learning 

**Title (ZH)**: 从问题解决到教学问题解决：通过强化学习使大语言模型与教学方法相alignment 

**Authors**: David Dinucu-Jianu, Jakub Macina, Nico Daheim, Ido Hakimi, Iryna Gurevych, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2505.15607)  

**Abstract**: Large language models (LLMs) can transform education, but their optimization for direct question-answering often undermines effective pedagogy which requires strategically withholding answers. To mitigate this, we propose an online reinforcement learning (RL)-based alignment framework that can quickly adapt LLMs into effective tutors using simulated student-tutor interactions by emphasizing pedagogical quality and guided problem-solving over simply giving away answers. We use our method to train a 7B parameter tutor model without human annotations which reaches similar performance to larger proprietary models like LearnLM. We introduce a controllable reward weighting to balance pedagogical support and student solving accuracy, allowing us to trace the Pareto frontier between these two objectives. Our models better preserve reasoning capabilities than single-turn SFT baselines and can optionally enhance interpretability through thinking tags that expose the model's instructional planning. 

**Abstract (ZH)**: 大型语言模型（LLMs）可以变革教育，但它们对直接问答的优化往往会削弱有效教学所需的战略性保留答案。为解决这一问题，我们提出了一种基于在线强化学习（RL）的对齐框架，该框架可以通过强调教学质量和引导性问题求解，快速将LLMs转换为有效的辅导工具，而不仅仅是直接给出答案。我们使用该方法训练了一个不依赖人类标注的7B参数辅导模型，其性能接近或堪比大型私有模型如LearnLM。我们引入了一种可控的奖励权重调整方法，以平衡教学支持和学生求解准确性，从而能够追踪这两项目标之间的帕累托前沿。我们的模型在保持推理能力方面优于单轮SFT基线，并且可以选择通过思维标签增强解释性，以揭示模型的教学规划。 

---
# Exploring LLM-Generated Feedback for Economics Essays: How Teaching Assistants Evaluate and Envision Its Use 

**Title (ZH)**: 探索LLM生成的反馈经济学论文评语：教学助理的评价及其使用展望 

**Authors**: Xinyi Lu, Aditya Mahesh, Zejia Shen, Mitchell Dudley, Larissa Sano, Xu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15596)  

**Abstract**: This project examines the prospect of using AI-generated feedback as suggestions to expedite and enhance human instructors' feedback provision. In particular, we focus on understanding the teaching assistants' perspectives on the quality of AI-generated feedback and how they may or may not utilize AI feedback in their own workflows. We situate our work in a foundational college Economics class, which has frequent short essay assignments. We developed an LLM-powered feedback engine that generates feedback on students' essays based on grading rubrics used by the teaching assistants (TAs). To ensure that TAs can meaningfully critique and engage with the AI feedback, we had them complete their regular grading jobs. For a randomly selected set of essays that they had graded, we used our feedback engine to generate feedback and displayed the feedback as in-text comments in a Word document. We then performed think-aloud studies with 5 TAs over 20 1-hour sessions to have them evaluate the AI feedback, contrast the AI feedback with their handwritten feedback, and share how they envision using the AI feedback if they were offered as suggestions. The study highlights the importance of providing detailed rubrics for AI to generate high-quality feedback for knowledge-intensive essays. TAs considered that using AI feedback as suggestions during their grading could expedite grading, enhance consistency, and improve overall feedback quality. We discuss the importance of decomposing the feedback generation task into steps and presenting intermediate results, in order for TAs to use the AI feedback. 

**Abstract (ZH)**: 本项目考察使用AI生成的反馈作为建议以加速和提升人类教师反馈提供的可能性。特别地，我们关注教学助手对AI生成反馈质量的看法，以及他们在工作流程中是否会或不会利用AI反馈。我们将工作置于一个基础大学经济学课程中，该课程有频繁的短 essays 作业。我们开发了一个基于大语言模型的反馈引擎，根据教学助手（TAs）使用的评分标准为学生论文生成反馈。为了确保教学助手能够有意义地批判和参与AI反馈，我们让他们完成了常规的批改工作。对他们的部分批改作业，我们使用反馈引擎生成反馈，并在Word文档中作为文本评论显示。然后，我们在20次每场时长为1小时的口头思考研究中与5位教学助手对AI反馈进行评估，将AI反馈与手写反馈进行对比，并分享如果提供作为建议时他们如何使用AI反馈的设想。研究强调了为AI提供详细的评分标准以生成高质量知识密集型论文反馈的重要性。教学助手认为在批改作业时利用AI反馈作为建议可以加速批改、增强一致性和提高整体反馈质量。我们讨论了将反馈生成任务分解成步骤并展示中间结果的重要性，以便教学助手使用AI反馈。 

---
# DayDreamer at CQs-Gen 2025: Generating Critical Questions through Argument Scheme Completion 

**Title (ZH)**: DayDreamer在CQs-Gen 2025：通过论据方案完成生成关键问题 

**Authors**: Wendi Zhou, Ameer Saadat-Yazdi, Nadin Kökciyan  

**Link**: [PDF](https://arxiv.org/pdf/2505.15554)  

**Abstract**: Critical questions are essential resources to provoke critical thinking when encountering an argumentative text. We present our system for the Critical Questions Generation (CQs-Gen) Shared Task at ArgMining 2025. Our approach leverages large language models (LLMs) with chain-of-thought prompting to generate critical questions guided by Walton's argumentation schemes. For each input intervention, we conversationally prompt LLMs to instantiate the corresponding argument scheme template to first obtain structured arguments, and then generate relevant critical questions. Following this, we rank all the available critical questions by prompting LLMs to select the top 3 most helpful questions based on the original intervention text. This combination of structured argumentation theory and step-by-step reasoning enables the generation of contextually relevant and diverse critical questions. Our pipeline achieves competitive performance in the final test set, showing its potential to foster critical thinking given argumentative text and detect missing or uninformed claims. Code available at \href{this https URL}{DayDreamer}. 

**Abstract (ZH)**: 批判性问题生成对于在遇到论辩性文本时激发批判性思维至关重要。我们在ArgMining 2025的批判性问题生成（CQs-Gen）共享任务中介绍了我们的系统。我们的方法利用大型语言模型（LLMs）结合链式思维提示，生成由沃顿的论辩框架引导的批判性问题。对于每个输入干预，我们以对话性提示LLMs实例化相应的论辩框架模板，首先获得结构化论点，然后生成相关的批判性问题。随后，我们通过提示LLMs根据原始干预文本选择最有帮助的前3个批判性问题进行排名。这种结构化论辩理论与逐步推理的结合使生成上下文相关且多样的批判性问题成为可能。我们的管道在最终测试集上实现了竞争性的性能，显示出其在给定向辩性文本促进批判性思维和检测缺失或不充分论断方面的潜力。代码可在DayDreamer获取。 

---
# Social Bias in Popular Question-Answering Benchmarks 

**Title (ZH)**: 流行问答基准中的社会偏见 

**Authors**: Angelie Kraft, Judith Simon, Sonja Schimmler  

**Link**: [PDF](https://arxiv.org/pdf/2505.15553)  

**Abstract**: Question-answering (QA) and reading comprehension (RC) benchmarks are essential for assessing the capabilities of large language models (LLMs) in retrieving and reproducing knowledge. However, we demonstrate that popular QA and RC benchmarks are biased and do not cover questions about different demographics or regions in a representative way, potentially due to a lack of diversity of those involved in their creation. We perform a qualitative content analysis of 30 benchmark papers and a quantitative analysis of 20 respective benchmark datasets to learn (1) who is involved in the benchmark creation, (2) how social bias is addressed or prevented, and (3) whether the demographics of the creators and annotators correspond to particular biases in the content. Most analyzed benchmark papers provided insufficient information regarding the stakeholders involved in benchmark creation, particularly the annotators. Notably, just one of the benchmark papers explicitly reported measures taken to address social representation issues. Moreover, the data analysis revealed gender, religion, and geographic biases across a wide range of encyclopedic, commonsense, and scholarly benchmarks. More transparent and bias-aware QA and RC benchmark creation practices are needed to facilitate better scrutiny and incentivize the development of fairer LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）的知识检索与再现能力评估：基于问答（QA）和阅读理解（RC）基准的多样性与偏见分析 

---
# Evaluate Bias without Manual Test Sets: A Concept Representation Perspective for LLMs 

**Title (ZH)**: 无需手动测试集评估偏差：面向LLM的概念表示视角 

**Authors**: Lang Gao, Kaiyang Wan, Wei Liu, Chenxi Wang, Zirui Song, Zixiang Xu, Yanbo Wang, Veselin Stoyanov, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.15524)  

**Abstract**: Bias in Large Language Models (LLMs) significantly undermines their reliability and fairness. We focus on a common form of bias: when two reference concepts in the model's concept space, such as sentiment polarities (e.g., "positive" and "negative"), are asymmetrically correlated with a third, target concept, such as a reviewing aspect, the model exhibits unintended bias. For instance, the understanding of "food" should not skew toward any particular sentiment. Existing bias evaluation methods assess behavioral differences of LLMs by constructing labeled data for different social groups and measuring model responses across them, a process that requires substantial human effort and captures only a limited set of social concepts. To overcome these limitations, we propose BiasLens, a test-set-free bias analysis framework based on the structure of the model's vector space. BiasLens combines Concept Activation Vectors (CAVs) with Sparse Autoencoders (SAEs) to extract interpretable concept representations, and quantifies bias by measuring the variation in representational similarity between the target concept and each of the reference concepts. Even without labeled data, BiasLens shows strong agreement with traditional bias evaluation metrics (Spearman correlation r > 0.85). Moreover, BiasLens reveals forms of bias that are difficult to detect using existing methods. For example, in simulated clinical scenarios, a patient's insurance status can cause the LLM to produce biased diagnostic assessments. Overall, BiasLens offers a scalable, interpretable, and efficient paradigm for bias discovery, paving the way for improving fairness and transparency in LLMs. 

**Abstract (ZH)**: BiasLens：基于模型向量空间结构的无测试集偏见分析框架 

---
# Protoknowledge Shapes Behaviour of LLMs in Downstream Tasks: Memorization and Generalization with Knowledge Graphs 

**Title (ZH)**: Protoknowledge 影响大规模语言模型在下游任务中行为的方式：知识图谱中的记忆与泛化 

**Authors**: Federico Ranaldi, Andrea Zugarini, Leonardo Ranaldi, Fabio Massimo Zanzotto  

**Link**: [PDF](https://arxiv.org/pdf/2505.15501)  

**Abstract**: We introduce the concept of protoknowledge to formalize and measure how sequences of tokens encoding Knowledge Graphs are internalized during pretraining and utilized at inference time by Large Language Models (LLMs). Indeed, LLMs have demonstrated the ability to memorize vast amounts of token sequences during pretraining, and a central open question is how they leverage this memorization as reusable knowledge through generalization. We then categorize protoknowledge into lexical, hierarchical, and topological forms, varying on the type of knowledge that needs to be activated. We measure protoknowledge through Knowledge Activation Tasks (KATs), analyzing its general properties such as semantic bias. We then investigate the impact of protoknowledge on Text-to-SPARQL performance by varying prompting strategies depending on input conditions. To this end, we adopt a novel analysis framework that assesses whether model predictions align with the successful activation of the relevant protoknowledge for each query. This methodology provides a practical tool to explore Semantic-Level Data Contamination and serves as an effective strategy for Closed-Pretraining models. 

**Abstract (ZH)**: 我们将概念化并衡量编码知识图谱的令牌序列在预训练期间的内化及其在推理时的利用过程，引入protoknowledge的概念。我们进一步将protoknowledge分为词缀的、层次的和拓扑的形式，根据不同类型的需激活知识进行分类。我们通过知识激活任务（KATs）测量protoknowledge，并分析其语义偏差等基本特性。然后，我们通过根据输入条件变化提示策略来研究protoknowledge对文本到SPARQL性能的影响。为此，我们采用一种新型分析框架，评估模型预测是否与每个查询的相关protoknowledge的成功激活一致。该方法提供了一种实用工具来探索语义级数据污染，并作为闭合预训练模型的有效策略。 

---
# LFTF: Locating First and Then Fine-Tuning for Mitigating Gender Bias in Large Language Models 

**Title (ZH)**: LFTF: 首先定位然后微调以减轻大型语言模型中的性别偏见 

**Authors**: Zhanyue Qin, Yue Ding, Deyuan Liu, Qingbin Liu, Junxian Cai, Xi Chen, Zhiying Tu, Dianhui Chu, Cuiyun Gao, Dianbo Sui  

**Link**: [PDF](https://arxiv.org/pdf/2505.15475)  

**Abstract**: Nowadays, Large Language Models (LLMs) have attracted widespread attention due to their powerful performance. However, due to the unavoidable exposure to socially biased data during training, LLMs tend to exhibit social biases, particularly gender bias. To better explore and quantifying the degree of gender bias in LLMs, we propose a pair of datasets named GenBiasEval and GenHintEval, respectively. The GenBiasEval is responsible for evaluating the degree of gender bias in LLMs, accompanied by an evaluation metric named AFGB-Score (Absolutely Fair Gender Bias Score). Meanwhile, the GenHintEval is used to assess whether LLMs can provide responses consistent with prompts that contain gender hints, along with the accompanying evaluation metric UB-Score (UnBias Score). Besides, in order to mitigate gender bias in LLMs more effectively, we present the LFTF (Locating First and Then Fine-Tuning) this http URL algorithm first ranks specific LLM blocks by their relevance to gender bias in descending order using a metric called BMI (Block Mitigating Importance Score). Based on this ranking, the block most strongly associated with gender bias is then fine-tuned using a carefully designed loss function. Numerous experiments have shown that our proposed LFTF algorithm can significantly mitigate gender bias in LLMs while maintaining their general capabilities. 

**Abstract (ZH)**: 现今，大型语言模型（LLMs）由于其强大的性能而引起了广泛关注。然而，由于不可避免地会接触到社会偏见数据的训练过程，LLMs往往会表现出社会偏见，特别是性别偏见。为了更深入地探索和量化LLMs中的性别偏见程度，我们提出了两个数据集，分别命名为GenBiasEval和GenHintEval。GenBiasEval用于评估LLMs中的性别偏见程度，伴随有一个名为AFGB-Score（绝对公平性别偏见评分）的评估指标。同时，GenHintEval用于评估LLMs是否能够提供与含有性别暗示的提示相一致的响应，伴随有UB-Score（无偏评分）的评估指标。此外，为了更有效地减轻LLMs中的性别偏见，我们提出了LFTF（定位并调整）算法。该算法首先使用称为BMI（块减轻重要性评分）的指标按降序对特定的LLM块进行排序。基于此排序，在一个精心设计的损失函数基础上，调整与性别偏见关联最紧密的块。大量实验表明，我们提出的LFTF算法在显著减轻LLMs中的性别偏见的同时，能够保持其一般性能。 

---
# A Qualitative Investigation into LLM-Generated Multilingual Code Comments and Automatic Evaluation Metrics 

**Title (ZH)**: 对LLM生成的多语言代码注释进行的定性研究及自动评价指标探究 

**Authors**: Jonathan Katzy, Yongcheng Huang, Gopal-Raj Panchu, Maksym Ziemlewski, Paris Loizides, Sander Vermeulen, Arie van Deursen, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2505.15469)  

**Abstract**: Large Language Models are essential coding assistants, yet their training is predominantly English-centric. In this study, we evaluate the performance of code language models in non-English contexts, identifying challenges in their adoption and integration into multilingual workflows. We conduct an open-coding study to analyze errors in code comments generated by five state-of-the-art code models, CodeGemma, CodeLlama, CodeQwen1.5, GraniteCode, and StarCoder2 across five natural languages: Chinese, Dutch, English, Greek, and Polish. Our study yields a dataset of 12,500 labeled generations, which we publicly release. We then assess the reliability of standard metrics in capturing comment \textit{correctness} across languages and evaluate their trustworthiness as judgment criteria. Through our open-coding investigation, we identified a taxonomy of 26 distinct error categories in model-generated code comments. They highlight variations in language cohesion, informativeness, and syntax adherence across different natural languages. Our analysis shows that, while these models frequently produce partially correct comments, modern neural metrics fail to reliably differentiate meaningful completions from random noise. Notably, the significant score overlap between expert-rated correct and incorrect comments calls into question the effectiveness of these metrics in assessing generated comments. 

**Abstract (ZH)**: 大型语言模型是重要的编码助手，但它们的训练主要以英语为中心。本研究评估了代码语言模型在非英语环境中的性能，识别其在多语言工作流程中的采用和集成所面临的挑战。我们进行了一项开放编码研究，分析了五种最新的代码模型CodeGemma、CodeLlama、CodeQwen1.5、GraniteCode和StarCoder2生成的代码注释错误，涵盖了五种自然语言：中文、荷兰语、英语、希腊语和波兰语。我们的研究生成了一个包含12,500个标注生成的语料库，并公开发布。然后我们评估了标准指标在跨语言捕捉注释的准确性的可靠性，并评估它们作为判断标准的信任度。通过我们的开放编码研究，我们识别出26个模型生成代码注释的不同错误类别，突显了不同自然语言在连贯性、信息量和语法遵守方面的差异。我们的分析显示，尽管这些模型经常生成部分正确的注释，但现代神经指标无法可靠地区分有意义的完成和随机噪声。值得注意的是，专家评级正确和不正确注释之间显著的得分重叠引发了这些指标评估生成注释有效性的问题。 

---
# Joint Flashback Adaptation for Forgetting-Resistant Instruction Tuning 

**Title (ZH)**: 遗忘 resistant 指令调优的联合 flashback 调整 

**Authors**: Yukun Zhao, Lingyong Yan, Zhenyang Li, Shuaiqiang Wang, Zhumin Chen, Zhaochun Ren, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.15467)  

**Abstract**: Large language models have achieved remarkable success in various tasks. However, it is challenging for them to learn new tasks incrementally due to catastrophic forgetting. Existing approaches rely on experience replay, optimization constraints, or task differentiation, which encounter strict limitations in real-world scenarios. To address these issues, we propose Joint Flashback Adaptation. We first introduce flashbacks -- a limited number of prompts from old tasks -- when adapting to new tasks and constrain the deviations of the model outputs compared to the original one. We then interpolate latent tasks between flashbacks and new tasks to enable jointly learning relevant latent tasks, new tasks, and flashbacks, alleviating data sparsity in flashbacks and facilitating knowledge sharing for smooth adaptation. Our method requires only a limited number of flashbacks without access to the replay data and is task-agnostic. We conduct extensive experiments on state-of-the-art large language models across 1000+ instruction-following tasks, arithmetic reasoning tasks, and general reasoning tasks. The results demonstrate the superior performance of our method in improving generalization on new tasks and reducing forgetting in old tasks. 

**Abstract (ZH)**: 大型语言模型在各类任务中取得了显著成功，但由于灾难性遗忘，它们在增量学习新任务方面面临挑战。现有方法依赖经验回放、优化约束或任务差异化，但在实际场景中会遇到严格限制。为解决这些问题，我们提出联合回溯适应方法。我们首先在适应新任务时引入回溯——来自旧任务的一小部分提示，限制模型输出与原始输出的偏差。然后，我们在回溯和新任务之间插值潜在任务，以实现潜在任务、新任务和回溯的联合学习，减少回溯中的数据稀疏性并促进知识共享，以便平滑适应。我们的方法仅需少量回溯，无需访问回放数据且无任务依赖性。我们在1000多个指令跟随任务、算术推理任务和一般推理任务上的先进大型语言模型上进行了广泛实验。结果表明，我们的方法在提高新任务泛化能力和减少旧任务遗忘方面表现出优越性能。 

---
# Single LLM, Multiple Roles: A Unified Retrieval-Augmented Generation Framework Using Role-Specific Token Optimization 

**Title (ZH)**: 单一LLM，多种角色：基于角色特定标记优化的统一检索增强生成框架 

**Authors**: Yutao Zhu, Jiajie Jin, Hongjin Qian, Zheng Liu, Zhicheng Dou, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.15444)  

**Abstract**: Existing studies have optimized retrieval-augmented generation (RAG) across various sub-tasks, such as query understanding and retrieval refinement, but integrating these optimizations into a unified framework remains challenging. To tackle this problem, this work proposes RoleRAG, a unified RAG framework that achieves efficient multi-task processing through role-specific token optimization. RoleRAG comprises six modules, each handling a specific sub-task within the RAG process. Additionally, we introduce a query graph to represent the decomposition of the query, which can be dynamically resolved according to the decomposing state. All modules are driven by the same underlying LLM, distinguished by task-specific role tokens that are individually optimized. This design allows RoleRAG to dynamically activate different modules within a single LLM instance, thereby streamlining deployment and reducing resource consumption. Experimental results on five open-domain question-answering datasets demonstrate the effectiveness, generalizability, and flexibility of our framework. 

**Abstract (ZH)**: 现有的研究已在各种子任务（如查询理解及检索精炼）上优化了检索增强生成（RAG）模型，但将这些优化整合到一个统一框架中仍具有挑战性。为解决这一问题，本工作提出了一种名为RoleRAG的统一RAG框架，通过角色特定的.token优化实现高效的多任务处理。RoleRAG包含了六个模块，每个模块负责RAG过程中的一个特定子任务。此外，我们引入了一个查询图来表示查询的分解，并可根据分解状态动态解决。所有模块由同一个底层LLM驱动，不同任务的角色标记分别优化。该设计使RoleRAG能够在单个LLM实例中动态激活不同的模块，从而简化部署并减少资源消耗。在五个开放领域问答数据集上的实验结果证明了该框架的有效性、泛化能力和灵活性。 

---
# Set-LLM: A Permutation-Invariant LLM 

**Title (ZH)**: Set-LLM: 一个不变排列的大语言模型 

**Authors**: Beni Egressy, Jan Stühmer  

**Link**: [PDF](https://arxiv.org/pdf/2505.15433)  

**Abstract**: While large language models (LLMs) demonstrate impressive capabilities across numerous applications, their robustness remains a critical concern. This paper is motivated by a specific vulnerability: the order sensitivity of LLMs. This vulnerability manifests itself as the order bias observed when LLMs decide between possible options (for example, a preference for the first option) and the tendency of LLMs to provide different answers when options are reordered. The use cases for this scenario extend beyond the classical case of multiple-choice question answering to the use of LLMs as automated evaluators in AI pipelines, comparing output generated by different models. We introduce Set-LLM, a novel architectural adaptation for pretrained LLMs that enables the processing of mixed set-text inputs with permutation invariance guarantees. The adaptations involve a new attention mask and new positional encodings specifically designed for sets. We provide a theoretical proof of invariance and demonstrate through experiments that Set-LLM can be trained effectively, achieving comparable or improved performance and maintaining the runtime of the original model, while eliminating order sensitivity. 

**Abstract (ZH)**: 大规模语言模型的顺序鲁棒性：Set-LLM架构的设计与实现 

---
# Responsible Diffusion Models via Constraining Text Embeddings within Safe Regions 

**Title (ZH)**: 负责任的Diffusion模型通过在安全区域内约束文本嵌入实现 

**Authors**: Zhiwen Li, Die Chen, Mingyuan Fan, Cen Chen, Yaliang Li, Yanhao Wang, Wenmeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.15427)  

**Abstract**: The remarkable ability of diffusion models to generate high-fidelity images has led to their widespread adoption. However, concerns have also arisen regarding their potential to produce Not Safe for Work (NSFW) content and exhibit social biases, hindering their practical use in real-world applications. In response to this challenge, prior work has focused on employing security filters to identify and exclude toxic text, or alternatively, fine-tuning pre-trained diffusion models to erase sensitive concepts. Unfortunately, existing methods struggle to achieve satisfactory performance in the sense that they can have a significant impact on the normal model output while still failing to prevent the generation of harmful content in some cases. In this paper, we propose a novel self-discovery approach to identifying a semantic direction vector in the embedding space to restrict text embedding within a safe region. Our method circumvents the need for correcting individual words within the input text and steers the entire text prompt towards a safe region in the embedding space, thereby enhancing model robustness against all possibly unsafe prompts. In addition, we employ Low-Rank Adaptation (LoRA) for semantic direction vector initialization to reduce the impact on the model performance for other semantics. Furthermore, our method can also be integrated with existing methods to improve their social responsibility. Extensive experiments on benchmark datasets demonstrate that our method can effectively reduce NSFW content and mitigate social bias generated by diffusion models compared to several state-of-the-art baselines. 

**Abstract (ZH)**: 扩散模型生成高保真图像的能力使其得到了广泛应用，但这也引发了对其可能生成不合适内容（NSFW内容）和社会偏见的担忧，阻碍了其在实际应用中的使用。为应对这一挑战，现有工作主要通过使用安全过滤器识别和排除有毒文本，或者微调预训练扩散模型以删除敏感概念。然而，现有方法在确保模型输出效果和防止生成有害内容方面难以达到满意的效果。本文提出了一种新的自发现方法，以在嵌入空间中识别语义方向向量，限制文本嵌入在安全区域。该方法绕过了对输入文本中个别词语进行修正的需求，而是引导整个文本提示向嵌入空间中的安全区域偏移，从而增强模型对所有可能不安全提示的鲁棒性。此外，我们采用低秩适应（LoRA）来初始化语义方向向量，以减少对其他语义性能的影响。同时，我们的方法还可以与现有方法集成，以提高其社会责任感。在基准数据集上的广泛实验表明，与几种最先进的基线方法相比，我们的方法可以有效减少NSFW内容并减轻扩散模型生成的社会偏见。 

---
# Silent Leaks: Implicit Knowledge Extraction Attack on RAG Systems through Benign Queries 

**Title (ZH)**: 静默泄露：通过良性查询对RAG系统进行隐性知识提取攻击 

**Authors**: Yuhao Wang, Wenjie Qu, Yanze Jiang, Zichen Liu, Yue Liu, Shengfang Zhai, Yinpeng Dong, Jiaheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15420)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance large language models (LLMs) by incorporating external knowledge bases, but they are vulnerable to privacy risks from data extraction attacks. Existing extraction methods typically rely on malicious inputs such as prompt injection or jailbreaking, making them easily detectable via input- or output-level detection. In this paper, we introduce Implicit Knowledge Extraction Attack (IKEA), which conducts knowledge extraction on RAG systems through benign queries. IKEA first leverages anchor concepts to generate queries with the natural appearance, and then designs two mechanisms to lead to anchor concept thoroughly 'explore' the RAG's privacy knowledge: (1) Experience Reflection Sampling, which samples anchor concepts based on past query-response patterns to ensure the queries' relevance to RAG documents; (2) Trust Region Directed Mutation, which iteratively mutates anchor concepts under similarity constraints to further exploit the embedding space. Extensive experiments demonstrate IKEA's effectiveness under various defenses, surpassing baselines by over 80% in extraction efficiency and 90% in attack success rate. Moreover, the substitute RAG system built from IKEA's extractions consistently outperforms those based on baseline methods across multiple evaluation tasks, underscoring the significant privacy risk in RAG systems. 

**Abstract (ZH)**: 隐式知识提取攻击（IKEA）：通过良性查询增强的大语言模型中的隐私知识提取 

---
# Audio Jailbreak: An Open Comprehensive Benchmark for Jailbreaking Large Audio-Language Models 

**Title (ZH)**: 音频脱狱：面向大型音频-语言模型脱狱的开放综合基准 

**Authors**: Zirui Song, Qian Jiang, Mingxuan Cui, Mingzhe Li, Lang Gao, Zeyu Zhang, Zixiang Xu, Yanbo Wang, Chenxi Wang, Guangxian Ouyang, Zhenhao Chen, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.15406)  

**Abstract**: The rise of Large Audio Language Models (LAMs) brings both potential and risks, as their audio outputs may contain harmful or unethical content. However, current research lacks a systematic, quantitative evaluation of LAM safety especially against jailbreak attacks, which are challenging due to the temporal and semantic nature of speech. To bridge this gap, we introduce AJailBench, the first benchmark specifically designed to evaluate jailbreak vulnerabilities in LAMs. We begin by constructing AJailBench-Base, a dataset of 1,495 adversarial audio prompts spanning 10 policy-violating categories, converted from textual jailbreak attacks using realistic text to speech synthesis. Using this dataset, we evaluate several state-of-the-art LAMs and reveal that none exhibit consistent robustness across attacks. To further strengthen jailbreak testing and simulate more realistic attack conditions, we propose a method to generate dynamic adversarial variants. Our Audio Perturbation Toolkit (APT) applies targeted distortions across time, frequency, and amplitude domains. To preserve the original jailbreak intent, we enforce a semantic consistency constraint and employ Bayesian optimization to efficiently search for perturbations that are both subtle and highly effective. This results in AJailBench-APT, an extended dataset of optimized adversarial audio samples. Our findings demonstrate that even small, semantically preserved perturbations can significantly reduce the safety performance of leading LAMs, underscoring the need for more robust and semantically aware defense mechanisms. 

**Abstract (ZH)**: LAMs兴起带来的机遇与风险：AJailBenchbenchmark及其应用 

---
# RePPL: Recalibrating Perplexity by Uncertainty in Semantic Propagation and Language Generation for Explainable QA Hallucination Detection 

**Title (ZH)**: RePPL：基于语义传播和语言生成不确定性调整困惑度的可解释QA幻觉检测 

**Authors**: Yiming Huang, Junyan Zhang, Zihao Wang, Biquan Bie, Xuming Hu, Yi R., Fung, Xinlei He  

**Link**: [PDF](https://arxiv.org/pdf/2505.15386)  

**Abstract**: Large Language Models (LLMs) have become powerful, but hallucinations remain a vital obstacle to their trustworthy use. While previous works improved the capability of hallucination detection by measuring uncertainty, they all lack the ability to explain the provenance behind why hallucinations occur, i.e., which part of the inputs tends to trigger hallucinations. Recent works on the prompt attack indicate that uncertainty exists in semantic propagation, where attention mechanisms gradually fuse local token information into high-level semantics across layers. Meanwhile, uncertainty also emerges in language generation, due to its probability-based selection of high-level semantics for sampled generations. Based on that, we propose RePPL to recalibrate uncertainty measurement by these two aspects, which dispatches explainable uncertainty scores to each token and aggregates in Perplexity-style Log-Average form as total score. Experiments show that our method achieves the best comprehensive detection performance across various QA datasets on advanced models (average AUC of 0.833), and our method is capable of producing token-level uncertainty scores as explanations for the hallucination. Leveraging these scores, we preliminarily find the chaotic pattern of hallucination and showcase its promising usage. 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为强大的工具，但幻觉仍然是其可靠使用的重要障碍。虽然之前的著作通过测量不确定性改善了幻觉检测能力，但它们缺乏解释幻觉发生根源的能力，即哪些输入部分容易引发幻觉。最近关于提示攻击的研究表明，在语义传播过程中，注意力机制逐渐将局部词元信息融合成高层语义，同时，由于基于概率的选择高层语义进行采样生成，不确定性也在语言生成中显现。基于此，我们提出RePPL来从这两个方面重新校准不确定性测量，为每个词元分配可解释的不确定性分数，并以困惑度风格的对数平均形式汇总为总分。实验表明，我们的方法在高级模型的各种QA数据集上实现了最佳综合检测性能（平均AUC为0.833），并且我们的方法能够生成词元级别的不确定性分数作为幻觉的解释。利用这些分数，我们初步揭示了幻觉的混乱模式，并展示了其潜在的应用价值。 

---
# Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors 

**Title (ZH)**: 你的语言模型能在不知不觉中模仿人类写作：对比重述对生成文本检测器的攻击 

**Authors**: Hao Fang, Jiawei Kong, Tianqu Zhuang, Yixiang Qiu, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Yaowei Wang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15337)  

**Abstract**: The misuse of large language models (LLMs), such as academic plagiarism, has driven the development of detectors to identify LLM-generated texts. To bypass these detectors, paraphrase attacks have emerged to purposely rewrite these texts to evade detection. Despite the success, existing methods require substantial data and computational budgets to train a specialized paraphraser, and their attack efficacy greatly reduces when faced with advanced detection algorithms. To address this, we propose \textbf{Co}ntrastive \textbf{P}araphrase \textbf{A}ttack (CoPA), a training-free method that effectively deceives text detectors using off-the-shelf LLMs. The first step is to carefully craft instructions that encourage LLMs to produce more human-like texts. Nonetheless, we observe that the inherent statistical biases of LLMs can still result in some generated texts carrying certain machine-like attributes that can be captured by detectors. To overcome this, CoPA constructs an auxiliary machine-like word distribution as a contrast to the human-like distribution generated by the LLM. By subtracting the machine-like patterns from the human-like distribution during the decoding process, CoPA is able to produce sentences that are less discernible by text detectors. Our theoretical analysis suggests the superiority of the proposed attack. Extensive experiments validate the effectiveness of CoPA in fooling text detectors across various scenarios. 

**Abstract (ZH)**: 基于现成大语言模型的Contrastive Paraphrase Attack (CoPA) 

---
# Trajectory Bellman Residual Minimization: A Simple Value-Based Method for LLM Reasoning 

**Title (ZH)**: 轨迹贝尔曼残差最小化：一种简单的基于值的方法用于LLM推理 

**Authors**: Yurun Yuan, Fan Chen, Zeyu Jia, Alexander Rakhlin, Tengyang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.15311)  

**Abstract**: Policy-based methods currently dominate reinforcement learning (RL) pipelines for large language model (LLM) reasoning, leaving value-based approaches largely unexplored. We revisit the classical paradigm of Bellman Residual Minimization and introduce Trajectory Bellman Residual Minimization (TBRM), an algorithm that naturally adapts this idea to LLMs, yielding a simple yet effective off-policy algorithm that optimizes a single trajectory-level Bellman objective using the model's own logits as $Q$-values. TBRM removes the need for critics, importance-sampling ratios, or clipping, and operates with only one rollout per prompt. We prove convergence to the near-optimal KL-regularized policy from arbitrary off-policy data via an improved change-of-trajectory-measure analysis. Experiments on standard mathematical-reasoning benchmarks show that TBRM consistently outperforms policy-based baselines, like PPO and GRPO, with comparable or lower computational and memory overhead. Our results indicate that value-based RL might be a principled and efficient alternative for enhancing reasoning capabilities in LLMs. 

**Abstract (ZH)**: 基于策略的方法目前主导了大规模语言模型（LLM）推理的强化学习（RL）流水线，价值基础的方法则 largely unexplored 基本未被探索。我们重访贝尔曼残差最小化的基本框架，提出轨迹贝尔曼残差最小化（TBRM），一种自然适应 LLM 的算法，生成一种简单而有效的离策优化算法，利用模型自身的 logit 作为 $Q$ 值优化单条轨迹级别贝尔曼目标。TBRM 去掉了批评家、重要性采样比率或裁剪的需求，且仅需一次滚出自每一个提示。我们通过改进的轨迹测度变化分析证明，TBRM 从任意离策数据中可收敛到近最优的 KL 正则化策略。标准的数学推理基准实验表明，TBRM 在与策略基础基线（如 PPO 和 GRPO）媲美的计算和内存开销条件下，始终表现出更优的效果。我们的结果表明，价值基础的强化学习可能是增强 LLM 推理能力的一种原理上合理且高效的替代方案。 

---
# Multiple Weaks Win Single Strong: Large Language Models Ensemble Weak Reinforcement Learning Agents into a Supreme One 

**Title (ZH)**: 多弱者胜一强：大型语言模型将弱强化学习代理ensemble成一个至高无上者 

**Authors**: Yiwen Song, Qianyue Hao, Qingmin Liao, Jian Yuan, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.15306)  

**Abstract**: Model ensemble is a useful approach in reinforcement learning (RL) for training effective agents. Despite wide success of RL, training effective agents remains difficult due to the multitude of factors requiring careful tuning, such as algorithm selection, hyperparameter settings, and even random seed choices, all of which can significantly influence an agent's performance. Model ensemble helps overcome this challenge by combining multiple weak agents into a single, more powerful one, enhancing overall performance. However, existing ensemble methods, such as majority voting and Boltzmann addition, are designed as fixed strategies and lack a semantic understanding of specific tasks, limiting their adaptability and effectiveness. To address this, we propose LLM-Ens, a novel approach that enhances RL model ensemble with task-specific semantic understandings driven by large language models (LLMs). Given a task, we first design an LLM to categorize states in this task into distinct 'situations', incorporating high-level descriptions of the task conditions. Then, we statistically analyze the strengths and weaknesses of each individual agent to be used in the ensemble in each situation. During the inference time, LLM-Ens dynamically identifies the changing task situation and switches to the agent that performs best in the current situation, ensuring dynamic model selection in the evolving task condition. Our approach is designed to be compatible with agents trained with different random seeds, hyperparameter settings, and various RL algorithms. Extensive experiments on the Atari benchmark show that LLM-Ens significantly improves the RL model ensemble, surpassing well-known baselines by up to 20.9%. For reproducibility, our code is open-source at this https URL. 

**Abstract (ZH)**: LLM驱动的强化学习模型集成方法：基于特定任务语义的理解（LLM-Ens） 

---
# Scaling Diffusion Transformers Efficiently via $μ$P 

**Title (ZH)**: 通过μP高效缩放扩散变换器 

**Authors**: Chenyu Zheng, Xinyu Zhang, Rongzhen Wang, Wei Huang, Zhi Tian, Weilin Huang, Jun Zhu, Chongxuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.15270)  

**Abstract**: Diffusion Transformers have emerged as the foundation for vision generative models, but their scalability is limited by the high cost of hyperparameter (HP) tuning at large scales. Recently, Maximal Update Parametrization ($\mu$P) was proposed for vanilla Transformers, which enables stable HP transfer from small to large language models, and dramatically reduces tuning costs. However, it remains unclear whether $\mu$P of vanilla Transformers extends to diffusion Transformers, which differ architecturally and objectively. In this work, we generalize standard $\mu$P to diffusion Transformers and validate its effectiveness through large-scale experiments. First, we rigorously prove that $\mu$P of mainstream diffusion Transformers, including DiT, U-ViT, PixArt-$\alpha$, and MMDiT, aligns with that of the vanilla Transformer, enabling the direct application of existing $\mu$P methodologies. Leveraging this result, we systematically demonstrate that DiT-$\mu$P enjoys robust HP transferability. Notably, DiT-XL-2-$\mu$P with transferred learning rate achieves 2.9 times faster convergence than the original DiT-XL-2. Finally, we validate the effectiveness of $\mu$P on text-to-image generation by scaling PixArt-$\alpha$ from 0.04B to 0.61B and MMDiT from 0.18B to 18B. In both cases, models under $\mu$P outperform their respective baselines while requiring small tuning cost, only 5.5% of one training run for PixArt-$\alpha$ and 3% of consumption by human experts for MMDiT-18B. These results establish $\mu$P as a principled and efficient framework for scaling diffusion Transformers. 

**Abstract (ZH)**: Maximal Update Parametrization for Scaling Diffusion Transformers 

---
# Blind Spot Navigation: Evolutionary Discovery of Sensitive Semantic Concepts for LVLMs 

**Title (ZH)**: 盲点导航：面向LVLMs的敏感语义概念进化发现 

**Authors**: Zihao Pan, Yu Tong, Weibin Wu, Jingyi Wang, Lifeng Chen, Zhe Zhao, Jiajia Wei, Yitong Qiao, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.15265)  

**Abstract**: Adversarial attacks aim to generate malicious inputs that mislead deep models, but beyond causing model failure, they cannot provide certain interpretable information such as ``\textit{What content in inputs make models more likely to fail?}'' However, this information is crucial for researchers to specifically improve model robustness. Recent research suggests that models may be particularly sensitive to certain semantics in visual inputs (such as ``wet,'' ``foggy''), making them prone to errors. Inspired by this, in this paper we conducted the first exploration on large vision-language models (LVLMs) and found that LVLMs indeed are susceptible to hallucinations and various errors when facing specific semantic concepts in images. To efficiently search for these sensitive concepts, we integrated large language models (LLMs) and text-to-image (T2I) models to propose a novel semantic evolution framework. Randomly initialized semantic concepts undergo LLM-based crossover and mutation operations to form image descriptions, which are then converted by T2I models into visual inputs for LVLMs. The task-specific performance of LVLMs on each input is quantified as fitness scores for the involved semantics and serves as reward signals to further guide LLMs in exploring concepts that induce LVLMs. Extensive experiments on seven mainstream LVLMs and two multimodal tasks demonstrate the effectiveness of our method. Additionally, we provide interesting findings about the sensitive semantics of LVLMs, aiming to inspire further in-depth research. 

**Abstract (ZH)**: 对抗攻击旨在生成恶意输入以误导深度模型，但这些攻击无法提供某些可解析的信息，如“是什么内容使得模型更有可能出错？”然而，这些信息对于研究人员具体提高模型鲁棒性至关重要。近期研究表明，模型可能特别容易对视觉输入中的某些语义（如“湿润”、“雾蒙蒙”）敏感，从而使它们容易出错。受此启发，本文首次探讨了大型视觉-语言模型（LVLMs），发现LVLMs在面对图像中特定语义概念时确实容易产生幻觉并出现各种错误。为了高效地搜索这些敏感概念，我们将大型语言模型（LLMs）和文本到图像（T2I）模型集成到一个新颖的语义进化框架中。随机初始化的语义概念通过LLM基于的交叉和变异操作生成图像描述，然后通过T2I模型转换为视觉输入供LVLMs使用。LVLMs在每个输入上的任务特定性能被量化为涉及语义的适应度分数，并作为奖励信号进一步引导LLMs探索导致LVLMs出错的概念。在七个主流LVLMs和两个多模态任务上的广泛实验表明了本方法的有效性。此外，我们提供了关于LVLMs敏感语义的有趣发现，旨在启发进一步深入的研究。 

---
# Towards Explainable Temporal Reasoning in Large Language Models: A Structure-Aware Generative Framework 

**Title (ZH)**: 面向可解释时间推理的大语言模型：一种结构感知生成框架 

**Authors**: Zihao Jiang, Ben Liu, Miao Peng, Wenjie Xu, Yao Xiao, Zhenyan Shan, Min Peng  

**Link**: [PDF](https://arxiv.org/pdf/2505.15245)  

**Abstract**: While large language models (LLMs) show great potential in temporal reasoning, most existing work focuses heavily on enhancing performance, often neglecting the explainable reasoning processes underlying the results. To address this gap, we introduce a comprehensive benchmark covering a wide range of temporal granularities, designed to systematically evaluate LLMs' capabilities in explainable temporal reasoning. Furthermore, our findings reveal that LLMs struggle to deliver convincing explanations when relying solely on textual information. To address challenge, we propose GETER, a novel structure-aware generative framework that integrates Graph structures with text for Explainable TEmporal Reasoning. Specifically, we first leverage temporal knowledge graphs to develop a temporal encoder that captures structural information for the query. Subsequently, we introduce a structure-text prefix adapter to map graph structure features into the text embedding space. Finally, LLMs generate explanation text by seamlessly integrating the soft graph token with instruction-tuning prompt tokens. Experimental results indicate that GETER achieves state-of-the-art performance while also demonstrating its effectiveness as well as strong generalization capabilities. Our dataset and code are available at this https URL. 

**Abstract (ZH)**: 虽然大型语言模型（LLMs）在时间推理方面显示出巨大的潜力，但现有大多数工作集中在提升性能上，往往忽视了支撑结果的可解释推理过程。为弥补这一差距，我们提出一个全面的时间粒度基准，旨在系统性地评估LLMs在可解释时间推理方面的能力。此外，我们的研究发现LLMs在仅依赖文本信息时难以提供令人信服的解释。为此，我们提出GETER，一种新颖的结构感知生成框架，将图结构与文本结合用于可解释时间推理。具体地，我们利用时间知识图谱开发时间编码器以捕捉查询的结构信息。随后，我们引入结构-文本前缀适配器将图结构特征映射至文本嵌入空间。最后，LLMs通过无缝整合软图令牌与指令调优提示令牌生成解释文本。实验结果表明，GETER在性能上达到了最新水平，同时展示了其有效性及强大的泛化能力。我们的数据集和代码可在以下链接获取。 

---
# Adaptive Plan-Execute Framework for Smart Contract Security Auditing 

**Title (ZH)**: 智能合约安全审计的自适应计划-执行框架 

**Authors**: Zhiyuan Wei, Jing Sun, Zijian Zhang, Zhe Hou, Zixiao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.15242)  

**Abstract**: Large Language Models (LLMs) have shown great promise in code analysis and auditing; however, they still struggle with hallucinations and limited context-aware reasoning. We introduce SmartAuditFlow, a novel Plan-Execute framework that enhances smart contract security analysis through dynamic audit planning and structured execution. Unlike conventional LLM-based auditing approaches that follow fixed workflows and predefined steps, SmartAuditFlow dynamically generates and refines audit plans based on the unique characteristics of each smart contract. It continuously adjusts its auditing strategy in response to intermediate LLM outputs and newly detected vulnerabilities, ensuring a more adaptive and precise security assessment. The framework then executes these plans step by step, applying a structured reasoning process to enhance vulnerability detection accuracy while minimizing hallucinations and false positives. To further improve audit precision, SmartAuditFlow integrates iterative prompt optimization and external knowledge sources, such as static analysis tools and Retrieval-Augmented Generation (RAG). This ensures audit decisions are contextually informed and backed by real-world security knowledge, producing comprehensive security reports. Extensive evaluations across multiple benchmarks demonstrate that SmartAuditFlow outperforms existing methods, achieving 100 percent accuracy on common and critical vulnerabilities, 41.2 percent accuracy for comprehensive coverage of known smart contract weaknesses in real-world projects, and successfully identifying all 13 tested CVEs. These results highlight SmartAuditFlow's scalability, cost-effectiveness, and superior adaptability over traditional static analysis tools and contemporary LLM-based approaches, establishing it as a robust solution for automated smart contract auditing. 

**Abstract (ZH)**: SmartAuditFlow：一种用于智能合约安全分析的新型计划-执行框架 

---
# ReflAct: World-Grounded Decision Making in LLM Agents via Goal-State Reflection 

**Title (ZH)**: ReflAct：通过目标状态反思实现基于世界的决策制定在LLM代理中 

**Authors**: Jeonghye Kim, Sojeong Rhee, Minbeom Kim, Dohyung Kim, Sangmook Lee, Youngchul Sung, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.15182)  

**Abstract**: Recent advances in LLM agents have largely built on reasoning backbones like ReAct, which interleave thought and action in complex environments. However, ReAct often produces ungrounded or incoherent reasoning steps, leading to misalignment between the agent's actual state and goal. Our analysis finds that this stems from ReAct's inability to maintain consistent internal beliefs and goal alignment, causing compounding errors and hallucinations. To address this, we introduce ReflAct, a novel backbone that shifts reasoning from merely planning next actions to continuously reflecting on the agent's state relative to its goal. By explicitly grounding decisions in states and enforcing ongoing goal alignment, ReflAct dramatically improves strategic reliability. This design delivers substantial empirical gains: ReflAct surpasses ReAct by 27.7% on average, achieving a 93.3% success rate in ALFWorld. Notably, ReflAct even outperforms ReAct with added enhancement modules (e.g., Reflexion, WKM), showing that strengthening the core reasoning backbone is key to reliable agent performance. 

**Abstract (ZH)**: 最近在LLM代理方面的进展大多建立在像ReAct这样的推理骨干之上，这些骨干在复杂环境中交替进行思考和行动。然而，ReAct经常产生不着边际或不连贯的推理步骤，导致代理的实际状态与目标之间的不对齐。我们的分析发现，这是由于ReAct无法维持一致的内部信念和目标对齐，导致累积错误和幻觉的产生。为了解决这个问题，我们引入了ReflAct，这是一种新型的骨干，将推理的重心从仅仅规划下一个动作转移到持续反思代理的状态与目标之间的关系。通过明确地将决策基于状态并持续强化目标对齐，ReflAct显著提高了战略可靠性。这一设计带来了实质性的实证收益：ReflAct在平均值上超越了ReAct 27.7%，在ALFWorld中达到了93.3%的成功率。值得注意的是，即使在添加了增强模块（如Reflexion、WKM）的情况下，ReflAct仍然优于ReAct，显示了加强核心推理骨干对于可靠代理性能的重要性。 

---
# Prolonged Reasoning Is Not All You Need: Certainty-Based Adaptive Routing for Efficient LLM/MLLM Reasoning 

**Title (ZH)**: 长时间推理并非全靠而已：基于 certainty 的自适应路由以实现高效的 LL offense 联合推理 

**Authors**: Jinghui Lu, Haiyang Yu, Siliang Xu, Shiwei Ran, Guozhi Tang, Siqi Wang, Bin Shan, Teng Fu, Hao Feng, Jingqun Tang, Han Wang, Can Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15154)  

**Abstract**: Recent advancements in reasoning have significantly enhanced the capabilities of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) across diverse tasks. However, excessive reliance on chain-of-thought (CoT) reasoning can impair model performance and brings unnecessarily lengthened outputs, reducing efficiency. Our work reveals that prolonged reasoning does not universally improve accuracy and even degrade performance on simpler tasks. To address this, we propose Certainty-based Adaptive Reasoning (CAR), a novel framework that dynamically switches between short answers and long-form reasoning based on the model perplexity. CAR first generates a short answer and evaluates its perplexity, triggering reasoning only when the model exhibits low confidence (i.e., high perplexity). Experiments across diverse multimodal VQA/KIE benchmarks and text reasoning datasets show that CAR outperforms both short-answer and long-form reasoning approaches, striking an optimal balance between accuracy and efficiency. 

**Abstract (ZH)**: 近期在推理方面的进展显著增强了大语言模型（LLMs）和多模态大语言模型（MLLMs）在各种任务中的能力。然而，过度依赖链式思考（CoT）推理会损害模型性能并导致不必要的输出延长，降低效率。我们的工作揭示了长时间推理并不普遍提高准确性，甚至在简单任务上降低性能。为了解决这个问题，我们提出了基于 certainty 的自适应推理（CAR）框架，该框架会根据模型困惑度动态切换短答案和长形式推理。CAR 首先生成一个短答案并评估其困惑度，仅当模型表现出低置信度（即高困惑度）时才触发推理。实验结果显示，CAR 在多种多模态 VQA/KIE 基准和文本推理数据集中均优于短答案和长形式推理方法，实现了准确性和效率之间的最佳平衡。 

---
# BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms 

**Title (ZH)**: BanditSpec: 通过bandit算法实现自适应投机解码 

**Authors**: Yunlong Hou, Fengzhuo Zhang, Cunxiao Du, Xuan Zhang, Jiachun Pan, Tianyu Pang, Chao Du, Vincent Y. F. Tan, Zhuoran Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15141)  

**Abstract**: Speculative decoding has emerged as a popular method to accelerate the inference of Large Language Models (LLMs) while retaining their superior text generation performance. Previous methods either adopt a fixed speculative decoding configuration regardless of the prefix tokens, or train draft models in an offline or online manner to align them with the context. This paper proposes a training-free online learning framework to adaptively choose the configuration of the hyperparameters for speculative decoding as text is being generated. We first formulate this hyperparameter selection problem as a Multi-Armed Bandit problem and provide a general speculative decoding framework BanditSpec. Furthermore, two bandit-based hyperparameter selection algorithms, UCBSpec and EXP3Spec, are designed and analyzed in terms of a novel quantity, the stopping time regret. We upper bound this regret under both stochastic and adversarial reward settings. By deriving an information-theoretic impossibility result, it is shown that the regret performance of UCBSpec is optimal up to universal constants. Finally, extensive empirical experiments with LLaMA3 and Qwen2 demonstrate that our algorithms are effective compared to existing methods, and the throughput is close to the oracle best hyperparameter in simulated real-life LLM serving scenarios with diverse input prompts. 

**Abstract (ZH)**: 无训练在线学习框架以适应性选择 speculate 解码超参数以加速大型语言模型的推理 

---
# The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning 

**Title (ZH)**: 熵最小化在大规模语言模型推理中的出人意料的有效性 

**Authors**: Shivam Agarwal, Zimin Zhang, Lifan Yuan, Jiawei Han, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2505.15134)  

**Abstract**: Entropy minimization (EM) trains the model to concentrate even more probability mass on its most confident outputs. We show that this simple objective alone, without any labeled data, can substantially improve large language models' (LLMs) performance on challenging math, physics, and coding tasks. We explore three approaches: (1) EM-FT minimizes token-level entropy similarly to instruction finetuning, but on unlabeled outputs drawn from the model; (2) EM-RL: reinforcement learning with negative entropy as the only reward to maximize; (3) EM-INF: inference-time logit adjustment to reduce entropy without any training data or parameter updates. On Qwen-7B, EM-RL, without any labeled data, achieves comparable or better performance than strong RL baselines such as GRPO and RLOO that are trained on 60K labeled examples. Furthermore, EM-INF enables Qwen-32B to match or exceed the performance of proprietary models like GPT-4o, Claude 3 Opus, and Gemini 1.5 Pro on the challenging SciCode benchmark, while being 3x more efficient than self-consistency and sequential refinement. Our findings reveal that many pretrained LLMs possess previously underappreciated reasoning capabilities that can be effectively elicited through entropy minimization alone, without any labeled data or even any parameter updates. 

**Abstract (ZH)**: 熵最小化（EM）使模型更集中于其最自信的输出。我们证明，仅此简单的目标，在没有任何标注数据的情况下，可以显著提高大型语言模型（LLMs）在挑战性的数学、物理和编程任务上的性能。我们探索了三种方法：（1）EM-FT在未标注的模型输出上像指令微调一样最小化令牌级熵；（2）EM-RL：仅以负熵作为奖励的最大化强化学习；（3）EM-INF：推理时的logit调整以减少熵，无需任何训练数据或参数更新。在Qwen-7B中，EM-RL在没有任何标注数据的情况下，取得了与基于6万个标注样例训练的强RL基线（如GRPO和RLOO）相当或更好的性能。此外，EM-INF使Qwen-32B能够匹配或超越自产模型（如GPT-4o、Claude 3 Opus和Gemini 1.5 Pro）在具有挑战性的SciCode基准测试中的性能，且效率高出3倍于自我一致性与序列完善。我们的研究结果表明，许多预训练LLM拥有以前被低估的推理能力，这些能力可以通过熵最小化等简单的任务激活，无需标注数据或参数更新。 

---
# An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents 

**Title (ZH)**: 强化学习在推理-搜索交织的大型语言模型代理上的实证研究 

**Authors**: Bowen Jin, Jinsung Yoon, Priyanka Kargupta, Sercan O. Arik, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.15117)  

**Abstract**: Reinforcement learning (RL) has demonstrated strong potential in training large language models (LLMs) capable of complex reasoning for real-world problem solving. More recently, RL has been leveraged to create sophisticated LLM-based search agents that adeptly combine reasoning with search engine use. While the use of RL for training search agents is promising, the optimal design of such agents remains not fully understood. In particular, key factors -- such as (1) reward formulation, (2) the choice and characteristics of the underlying LLM, and (3) the role of the search engine in the RL process -- require further investigation. In this work, we conduct comprehensive empirical studies to systematically investigate these and offer actionable insights. We highlight several key findings: format rewards are effective in improving final performance, whereas intermediate retrieval rewards have limited impact; the scale and initialization of the LLM (general-purpose vs. reasoning-specialized) significantly influence RL outcomes; and the choice of search engine plays a critical role in shaping RL training dynamics and the robustness of the trained agent during inference. These establish important guidelines for successfully building and deploying LLM-based search agents in real-world applications. Code is available at this https URL. 

**Abstract (ZH)**: 强化学习（RL）在训练具备复杂推理能力的大语言模型（LLMs）以解决实际问题方面展现了强大的潜力。最近，RL被用于创建能够熟练结合推理与搜索引擎使用的精妙LLM搜索代理。尽管使用RL训练搜索代理具有前景，但这类代理的最佳设计仍不完全清晰。特别是，关键因素包括（1）奖励的制定，（2）基础大语言模型的选择及其特性，以及（3）搜索引擎在RL过程中的作用，都需要进一步研究。本研究通过综合的实证研究系统地探讨这些因素，并提供了实用的见解。我们突出了一些关键发现：格式奖励有助于提升最终性能，而中间检索奖励的影响则有限；大语言模型的规模和初始化（通用型 vs. 专门用于推理）对RL结果有显著影响；搜索引擎的选择对RL训练动态以及推理期间代理的鲁棒性起着决定性作用。这些研究建立了在实际应用中成功构建和部署基于LLM的搜索代理的重要指导方针。代码可在以下链接获得：this https URL。 

---
# A Risk Taxonomy for Evaluating AI-Powered Psychotherapy Agents 

**Title (ZH)**: AI赋能心理疗法代理的风险分类 

**Authors**: Ian Steenstra, Timothy W. Bickmore  

**Link**: [PDF](https://arxiv.org/pdf/2505.15108)  

**Abstract**: The proliferation of Large Language Models (LLMs) and Intelligent Virtual Agents acting as psychotherapists presents significant opportunities for expanding mental healthcare access. However, their deployment has also been linked to serious adverse outcomes, including user harm and suicide, facilitated by a lack of standardized evaluation methodologies capable of capturing the nuanced risks of therapeutic interaction. Current evaluation techniques lack the sensitivity to detect subtle changes in patient cognition and behavior during therapy sessions that may lead to subsequent decompensation. We introduce a novel risk taxonomy specifically designed for the systematic evaluation of conversational AI psychotherapists. Developed through an iterative process including review of the psychotherapy risk literature, qualitative interviews with clinical and legal experts, and alignment with established clinical criteria (e.g., DSM-5) and existing assessment tools (e.g., NEQ, UE-ATR), the taxonomy aims to provide a structured approach to identifying and assessing user/patient harms. We provide a high-level overview of this taxonomy, detailing its grounding, and discuss potential use cases. We discuss two use cases in detail: monitoring cognitive model-based risk factors during a counseling conversation to detect unsafe deviations, in both human-AI counseling sessions and in automated benchmarking of AI psychotherapists with simulated patients. The proposed taxonomy offers a foundational step towards establishing safer and more responsible innovation in the domain of AI-driven mental health support. 

**Abstract (ZH)**: 大型语言模型（LLMs）和充当心理咨询师的智能虚拟代理的 proliferations 为扩展心理健康服务的可及性带来了重要机会，但它们的应用也与严重的负面后果相关联，包括用户伤害和自杀，这些都是由于缺乏能够捕捉治疗互动复杂风险的标准评估方法。当前的评估技术无法捕捉到治疗会话中患者认知和行为的微妙变化，这些变化可能导致后续的病情恶化。我们介绍了一种新型的风险分类学，特别设计用于系统评估对话式人工智能心理咨询师。该分类学通过迭代过程开发，包括心理治疗风险文献的回顾、与临床和法律专家的质性访谈，以及与现有的临床标准（如DSM-5）和评估工具（如NEQ，UE-ATR）的对齐。该分类学旨在提供一种结构化的方法来识别和评估用户/患者伤害。我们提供了该分类学的高层次概述，详细说明其理论基础，并讨论潜在的应用案例。我们详细讨论了两个应用案例：在咨询对话中监控基于认知模型的风险因素，以检测不安全的偏差，无论是人类-人工智能咨询会话还是自动化的基于模拟患者的AI心理咨询评估。提出的风险分类学为我们朝着建立更安全和更负责任的人工智能驱动心理健康支持领域创新奠定了基础。 

---
# StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization 

**Title (ZH)**: StepSearch：通过逐步近端策略优化激发大规模语言模型的检索能力 

**Authors**: Ziliang Wang, Xuhui Zheng, Kang An, Cijun Ouyang, Jialu Cai, Yuhang Wang, Yichao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15107)  

**Abstract**: Efficient multi-hop reasoning requires Large Language Models (LLMs) based agents to acquire high-value external knowledge iteratively. Previous work has explored reinforcement learning (RL) to train LLMs to perform search-based document retrieval, achieving notable improvements in QA performance, but underperform on complex, multi-hop QA resulting from the sparse rewards from global signal only. To address this gap in existing research, we introduce StepSearch, a framework for search LLMs that trained with step-wise proximal policy optimization method. It consists of richer and more detailed intermediate search rewards and token-level process supervision based on information gain and redundancy penalties to better guide each search step. We constructed a fine-grained question-answering dataset containing sub-question-level search trajectories based on open source datasets through a set of data pipeline method. On standard multi-hop QA benchmarks, it significantly outperforms global-reward baselines, achieving 11.2% and 4.2% absolute improvements for 3B and 7B models over various search with RL baselines using only 19k training data, demonstrating the effectiveness of fine-grained, stepwise supervision in optimizing deep search LLMs. Our implementation is publicly available at this https URL. 

**Abstract (ZH)**: 高效多跳推理需要基于大规模语言模型的代理逐步迭代获取高价值外部知识。现有工作探索使用强化学习训练大规模语言模型进行基于搜索的文档检索，取得了显著的问答性能改进，但在从全球信号稀疏奖励中产生的复杂多跳问答任务上表现不佳。为弥补现有研究的这一差距，我们引入了StepSearch，一种基于逐步近端策略优化方法训练的搜索框架。它包含更丰富、更详细的中间搜索奖励和基于信息增益和冗余惩罚的token级别过程监督，以更好地引导每一步搜索。我们通过一套数据管道方法，根据开源数据集构建了一个粒度精细的问答数据集，包含子问题级别的搜索轨迹。在标准多跳问答基准测试上，它显著优于全局奖励基准，仅使用19,000条训练数据，3B和7B模型分别实现了11.2%和4.2%的绝对性能提升，证明了细致步骤监督在优化深层搜索大规模语言模型中的有效性。我们的实现可在以下网址获取：this https URL。 

---
# ThinkRec: Thinking-based recommendation via LLM 

**Title (ZH)**: 基于思维的推荐：通过大规模语言模型实现 

**Authors**: Qihang Yu, Kairui Fu, Shengyu Zhang, Zheqi Lv, Fan Wu, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15091)  

**Abstract**: Recent advances in large language models (LLMs) have enabled more semantic-aware recommendations through natural language generation. Existing LLM for recommendation (LLM4Rec) methods mostly operate in a System 1-like manner, relying on superficial features to match similar items based on click history, rather than reasoning through deeper behavioral logic. This often leads to superficial and erroneous recommendations. Motivated by this, we propose ThinkRec, a thinking-based framework that shifts LLM4Rec from System 1 to System 2 (rational system). Technically, ThinkRec introduces a thinking activation mechanism that augments item metadata with keyword summarization and injects synthetic reasoning traces, guiding the model to form interpretable reasoning chains that consist of analyzing interaction histories, identifying user preferences, and making decisions based on target items. On top of this, we propose an instance-wise expert fusion mechanism to reduce the reasoning difficulty. By dynamically assigning weights to expert models based on users' latent features, ThinkRec adapts its reasoning path to individual users, thereby enhancing precision and personalization. Extensive experiments on real-world datasets demonstrate that ThinkRec significantly improves the accuracy and interpretability of recommendations. Our implementations are available in anonymous Github: this https URL. 

**Abstract (ZH)**: Recent advances in大规模语言模型(LLMs)使通过自然语言生成实现更具语义意识的推荐成为可能。现有推荐中的大规模语言模型(LLM4Rec)方法主要以System 1（直觉系统）的方式运行，依赖于表面特征基于点击历史匹配相似项目，而不是通过更深层次的行为逻辑进行推理。这往往会导致肤浅且错误的推荐。鉴于此，我们提出了基于思考的框架ThinkRec，将LLM4Rec从System 1转变为System 2（理性系统）。技术上，ThinkRec引入了一种思考激活机制，通过关键词总结增强项目元数据，并注入合成的推理痕迹，引导模型形成可解释的推理链，包括分析交互历史、识别用户偏好并基于目标项目做出决策。在此基础上，我们提出了一种实例级专家融合机制以降低推理难度。通过根据用户潜在特征动态分配专家模型的权重，ThinkRec能够根据个体用户的需要调整推理路径，从而提高精准度和个性化。在真实世界数据集上的广泛实验表明，ThinkRec显著提高了推荐的准确性和可解释性。我们的实现已匿名发布在GitHub：this https URL。 

---
# DeFTX: Denoised Sparse Fine-Tuning for Zero-Shot Cross-Lingual Transfer 

**Title (ZH)**: DeFTX: 去噪稀疏微调在零-shot 跨语言迁移中的应用 

**Authors**: Sona Elza Simon, Preethi Jyothi  

**Link**: [PDF](https://arxiv.org/pdf/2505.15090)  

**Abstract**: Effective cross-lingual transfer remains a critical challenge in scaling the benefits of large language models from high-resource to low-resource languages. Towards this goal, prior studies have explored many approaches to combine task knowledge from task-specific data in a (high-resource) source language and language knowledge from unlabeled text in a (low-resource) target language. One notable approach proposed composable sparse fine-tuning (SFT) for cross-lingual transfer that learns task-specific and language-specific sparse masks to select a subset of the pretrained model's parameters that are further fine-tuned. These sparse fine-tuned vectors (SFTs) are subsequently composed with the pretrained model to facilitate zero-shot cross-lingual transfer to a task in a target language, using only task-specific data from a source language. These sparse masks for SFTs were identified using a simple magnitude-based pruning. In our work, we introduce DeFT-X, a novel composable SFT approach that denoises the weight matrices of a pretrained model before magnitude pruning using singular value decomposition, thus yielding more robust SFTs. We evaluate DeFT-X on a diverse set of extremely low-resource languages for sentiment classification (NusaX) and natural language inference (AmericasNLI) and demonstrate that it performs at par or outperforms SFT and other prominent cross-lingual transfer baselines. 

**Abstract (ZH)**: 有效的跨语言迁移仍然是将大型语言模型的优势从资源丰富语言扩展到资源贫乏语言的关键挑战。为了实现这一目标，先前的研究探索了许多方法，将特定任务的知识从资源丰富语言的源语言中的特定任务数据与资源贫乏语言的目标语言中的未标记文本的语言知识结合起来。其中一种 notable 的方法是可组合的稀疏微调（SFT），该方法学习特定任务和特定语言的稀疏掩码以选择预训练模型参数的子集，并进一步微调。这些稀疏微调向量（SFTs）随后与预训练模型组合，以便仅使用资源丰富语言中的特定任务数据，实现目标语言中的零样本跨语言迁移。这些 SFT 的稀疏掩码是使用简单的基于幅度的剪枝方法识别的。在我们的工作中，我们介绍了 DeFT-X，这是一种新颖的可组合 SFT 方法，在进行幅度剪枝之前，使用奇异值分解清理预训练模型的权重矩阵，从而生成更 robust 的 SFT。我们在情感分类（NusaX）和自然语言推断（AmericasNLI）等一系列极其资源贫乏的语言上评估了 DeFT-X，并证明它与 SFT 和其他主要的跨语言迁移基准方法性能相当或更好。 

---
# Leveraging Large Language Models for Command Injection Vulnerability Analysis in Python: An Empirical Study on Popular Open-Source Projects 

**Title (ZH)**: 利用大型语言模型分析Python中的命令注入漏洞：基于流行开源项目的实证研究 

**Authors**: Yuxuan Wang, Jingshu Chen, Qingyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15088)  

**Abstract**: Command injection vulnerabilities are a significant security threat in dynamic languages like Python, particularly in widely used open-source projects where security issues can have extensive impact. With the proven effectiveness of Large Language Models(LLMs) in code-related tasks, such as testing, researchers have explored their potential for vulnerabilities analysis. This study evaluates the potential of large language models (LLMs), such as GPT-4, as an alternative approach for automated testing for vulnerability detection. In particular, LLMs have demonstrated advanced contextual understanding and adaptability, making them promising candidates for identifying nuanced security vulnerabilities within code. To evaluate this potential, we applied LLM-based analysis to six high-profile GitHub projects-Django, Flask, TensorFlow, Scikit-learn, PyTorch, and Langchain-each with over 50,000 stars and extensive adoption across software development and academic research. Our analysis assesses both the strengths and limitations of LLMs in detecting command injection vulnerabilities, evaluating factors such as detection accuracy, efficiency, and practical integration into development workflows. In addition, we provide a comparative analysis of different LLM tools to identify those most suitable for security applications. Our findings offer guidance for developers and security researchers on leveraging LLMs as innovative and automated approaches to enhance software security. 

**Abstract (ZH)**: 大型语言模型在检测命令注入漏洞中的潜力：基于GitHub上广泛使用的六个高知名度项目的评估 

---
# Traveling Across Languages: Benchmarking Cross-Lingual Consistency in Multimodal LLMs 

**Title (ZH)**: 跨国界旅行：多模态LLM跨语言一致性Benchmark研究 

**Authors**: Hao Wang, Pinzhi Huang, Jihan Yang, Saining Xie, Daisuke Kawahara  

**Link**: [PDF](https://arxiv.org/pdf/2505.15075)  

**Abstract**: The rapid evolution of multimodal large language models (MLLMs) has significantly enhanced their real-world applications. However, achieving consistent performance across languages, especially when integrating cultural knowledge, remains a significant challenge. To better assess this issue, we introduce two new benchmarks: KnowRecall and VisRecall, which evaluate cross-lingual consistency in MLLMs. KnowRecall is a visual question answering benchmark designed to measure factual knowledge consistency in 15 languages, focusing on cultural and historical questions about global landmarks. VisRecall assesses visual memory consistency by asking models to describe landmark appearances in 9 languages without access to images. Experimental results reveal that state-of-the-art MLLMs, including proprietary ones, still struggle to achieve cross-lingual consistency. This underscores the need for more robust approaches that produce truly multilingual and culturally aware models. 

**Abstract (ZH)**: 多模态大型语言模型的快速进化显著增强了其实际应用能力。然而，在跨语言应用，尤其是在整合文化知识方面，实现一致的性能仍旧是一项重大挑战。为了更好地评估这一问题，我们引入了两个新的基准：KnowRecall和VisRecall，用于评估多模态大型语言模型的跨语言一致性。KnowRecall是一个视觉问答基准，旨在测量15种语言中的事实知识一致性，重点关注关于全球地标的文化和历史问题。VisRecall通过要求模型在没有图片访问的情况下描述9种语言中的地标外貌，评估视觉记忆一致性。实验结果表明，最先进的多模态大型语言模型，包括专有模型，仍然难以实现跨语言一致性。这强调了需要更加 robust的方法来生产真正多语言且文化意识强的模型。 

---
# DISCO Balances the Scales: Adaptive Domain- and Difficulty-Aware Reinforcement Learning on Imbalanced Data 

**Title (ZH)**: DISCO 调和天平：不平衡数据上的自适应领域和难度感知强化学习 

**Authors**: Yuhang Zhou, Jing Zhu, Shengyi Qian, Zhuokai Zhao, Xiyao Wang, Xiaoyu Liu, Ming Li, Paiheng Xu, Wei Ai, Furong Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15074)  

**Abstract**: Large Language Models (LLMs) are increasingly aligned with human preferences through Reinforcement Learning from Human Feedback (RLHF). Among RLHF methods, Group Relative Policy Optimization (GRPO) has gained attention for its simplicity and strong performance, notably eliminating the need for a learned value function. However, GRPO implicitly assumes a balanced domain distribution and uniform semantic alignment across groups - assumptions that rarely hold in real-world datasets. When applied to multi-domain, imbalanced data, GRPO disproportionately optimizes for dominant domains, neglecting underrepresented ones and resulting in poor generalization and fairness. We propose Domain-Informed Self-Consistency Policy Optimization (DISCO), a principled extension to GRPO that addresses inter-group imbalance with two key innovations. Domain-aware reward scaling counteracts frequency bias by reweighting optimization based on domain prevalence. Difficulty-aware reward scaling leverages prompt-level self-consistency to identify and prioritize uncertain prompts that offer greater learning value. Together, these strategies promote more equitable and effective policy learning across domains. Extensive experiments across multiple LLMs and skewed training distributions show that DISCO improves generalization, outperforms existing GRPO variants by 5% on Qwen3 models, and sets new state-of-the-art results on multi-domain alignment benchmarks. 

**Abstract (ZH)**: 基于域信息的自我一致性策略优化（DISCO）：一种解决不平衡域分布的原理性扩展 

---
# The Pursuit of Empathy: Evaluating Small Language Models for PTSD Dialogue Support 

**Title (ZH)**: 共情的追求：评估小型语言模型在 PTSD 对话支持中的效果 

**Authors**: Suhas BN, Yash Mahajan, Dominik Mattioli, Andrew M. Sherrill, Rosa I. Arriaga, Chris W. Wiese, Saeed Abdullah  

**Link**: [PDF](https://arxiv.org/pdf/2505.15065)  

**Abstract**: Can small language models with 0.5B to 5B parameters meaningfully engage in trauma-informed, empathetic dialogue for individuals with PTSD? We address this question by introducing TIDE, a dataset of 10,000 two-turn dialogues spanning 500 diverse PTSD client personas and grounded in a three-factor empathy model: emotion recognition, distress normalization, and supportive reflection. All scenarios and reference responses were reviewed for realism and trauma sensitivity by a clinical psychologist specializing in PTSD. We evaluate eight small language models before and after fine-tuning, comparing their outputs to a frontier model (Claude Sonnet 3.5). Our IRB-approved human evaluation and automatic metrics show that fine-tuning generally improves perceived empathy, but gains are highly scenario- and user-dependent, with smaller models facing an empathy ceiling. Demographic analysis shows older adults value distress validation and graduate-educated users prefer nuanced replies, while gender effects are minimal. We highlight the limitations of automatic metrics and the need for context- and user-aware system design. Our findings, along with the planned release of TIDE, provide a foundation for building safe, resource-efficient, and ethically sound empathetic AI to supplement, not replace, clinical mental health care. 

**Abstract (ZH)**: 0.5B至5B参数的小语言模型能否有意义地与 PTSD 患者进行创伤知情、共情对话？我们通过引入 TIDE 数据集来解答这个问题，该数据集包含 10,000 个双轮对话，覆盖了 500 种多样化的 PTSD 患者人设，并基于一种三因素共情模型：情绪识别、痛苦正常化和支持性反思。所有场景和参考回复均经专门研究 PTSD 的临床心理学家审核，确保其真实性和对创伤的敏感性。我们对八种小语言模型进行微调前后的评估，并将其输出与前沿模型（Claude Sonnet 3.5）进行对比。经 IRB 批准的人类评估和自动指标表明，微调通常能提高感知共情，但进步程度高度依赖于具体场景和用户，而小型模型则面临共情上限。人口统计分析显示，老年人更看重痛苦的验证，受过高等教育的用户更喜欢细致的回答，而性别影响则较小。我们强调自动指标的局限性，并强调需要情境和用户意识系统的设计。我们的发现，加上 TIDE 数据集的计划发布，为建立安全、资源高效且伦理合理的共情 AI 提供了基础，以补充而非替代临床心理健康护理。 

---
# Self-GIVE: Associative Thinking from Limited Structured Knowledge for Enhanced Large Language Model Reasoning 

**Title (ZH)**: Self-GIVE：从有限结构化知识中进行关联思考以增强大规模语言模型推理 

**Authors**: Jiashu He, Jinxuan Fan, Bowen Jiang, Ignacio Houine, Dan Roth, Alejandro Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2505.15062)  

**Abstract**: When addressing complex questions that require new information, people often associate the question with existing knowledge to derive a sensible answer. For instance, when evaluating whether melatonin aids insomnia, one might associate "hormones helping mental disorders" with "melatonin being a hormone and insomnia a mental disorder" to complete the reasoning. Large Language Models (LLMs) also require such associative thinking, particularly in resolving scientific inquiries when retrieved knowledge is insufficient and does not directly answer the question. Graph Inspired Veracity Extrapolation (GIVE) addresses this by using a knowledge graph (KG) to extrapolate structured knowledge. However, it involves the construction and pruning of many hypothetical triplets, which limits efficiency and generalizability. We propose Self-GIVE, a retrieve-RL framework that enhances LLMs with automatic associative thinking through reinforcement learning. Self-GIVE extracts structured information and entity sets to assist the model in linking to the queried concepts. We address GIVE's key limitations: (1) extensive LLM calls and token overhead for knowledge extrapolation, (2) difficulty in deploying on smaller LLMs (3B or 7B) due to complex instructions, and (3) inaccurate knowledge from LLM pruning. Specifically, after fine-tuning using self-GIVE with a 135 node UMLS KG, it improves the performance of the Qwen2.5 3B and 7B models by up to $\textbf{28.5%$\rightarrow$71.4%}$ and $\textbf{78.6$\rightarrow$90.5%}$ in samples $\textbf{unseen}$ in challenging biomedical QA tasks. In particular, Self-GIVE allows the 7B model to match or outperform GPT3.5 turbo with GIVE, while cutting token usage by over 90\%. Self-GIVE enhances the scalable integration of structured retrieval and reasoning with associative thinking. 

**Abstract (ZH)**: 基于图启发的真伪外推自构模型 

---
# PiFlow: Principle-aware Scientific Discovery with Multi-Agent Collaboration 

**Title (ZH)**: PiFlow: 原理导向的多Agent协作科学研究 

**Authors**: Yingming Pu, Tao Lin, Hongyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.15047)  

**Abstract**: Large Language Model (LLM)-based multi-agent systems (MAS) demonstrate remarkable potential for scientific discovery. Existing approaches, however, often automate scientific discovery using predefined workflows that lack rationality constraints. This often leads to aimless hypothesizing and a failure to consistently link hypotheses with evidence, thereby hindering systematic uncertainty reduction. Overcoming these limitations fundamentally requires systematic uncertainty reduction. We introduce \texttt{PiFlow}, an information-theoretical framework, treating automated scientific discovery as a structured uncertainty reduction problem guided by principles (e.g., scientific laws). In evaluations across three distinct scientific domains -- discovering nanomaterial structures, bio-molecules, and superconductor candidates with targeted properties -- our method significantly improves discovery efficiency, reflected by a 73.55\% increase in the Area Under the Curve (AUC) of property values versus exploration steps, and enhances solution quality by 94.06\% compared to a vanilla agent system. Overall, \texttt{PiFlow} serves as a Plug-and-Play method, establishing a novel paradigm shift in highly efficient automated scientific discovery, paving the way for more robust and accelerated AI-driven research. Code is publicly available at our \href{this https URL}{GitHub}. 

**Abstract (ZH)**: 基于大型语言模型（LLM）的多agents系统（MAS）在科学研究中展现出巨大的潜力。现有方法通常使用预定义的工作流来进行科学研究自动化，缺乏理性的约束，导致盲目假设和无法一致地将假设与证据关联起来，从而妨碍系统的不确定性减少。克服这些限制从根本上需要系统性的不确定性减少。我们提出了\texttt{PiFlow}，一种信息论框架，将自动科学研究视为由原理（例如，科学定律）指导的结构化不确定性减少问题。在涉及三种不同科学领域的评估中——纳米材料结构发现、生物分子发现以及具有目标性质的超导体候选物发现——我们的方法显著提高了发现效率，表现为面积下曲线（AUC）的属性值与探索步骤之间的AUC提高了73.55%，并且相较于传统的代理系统，解决方案质量提高了94.06%。总体而言，\texttt{PiFlow}作为一种即插即用方法，为高效自动科学研究建立了新的范式转变，为更稳健和快速的人工智能驱动研究铺平了道路。代码在我们的GitHub上公开可用。 

---
# Denoising Concept Vectors with Sparse Autoencoders for Improved Language Model Steering 

**Title (ZH)**: 使用稀疏自编码器去噪概念向量以改进语言模型导向 

**Authors**: Haiyan Zhao, Xuansheng Wu, Fan Yang, Bo Shen, Ninghao Liu, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.15038)  

**Abstract**: Linear Concept Vectors have proven effective for steering large language models (LLMs). While existing approaches like linear probing and difference-in-means derive these vectors from LLM hidden representations, diverse data introduces noises (i.e., irrelevant features) that challenge steering robustness. To address this, we propose Sparse Autoencoder-Denoised Concept Vectors (SDCV), which uses Sparse Autoencoders to filter out noisy features from hidden representations. When applied to linear probing and difference-in-means, our method improves their steering success rates. We validate our noise hypothesis through counterfactual experiments and feature visualizations. 

**Abstract (ZH)**: Sparse Autoencoder-Denoised Concept Vectors for Robust Steering of Large Language Models 

---
# RL Tango: Reinforcing Generator and Verifier Together for Language Reasoning 

**Title (ZH)**: RL �人才舞蹈：同时增强生成器和验证器的语言推理 

**Authors**: Kaiwen Zha, Zhengqi Gao, Maohao Shen, Zhang-Wei Hong, Duane S. Boning, Dina Katabi  

**Link**: [PDF](https://arxiv.org/pdf/2505.15034)  

**Abstract**: Reinforcement learning (RL) has recently emerged as a compelling approach for enhancing the reasoning capabilities of large language models (LLMs), where an LLM generator serves as a policy guided by a verifier (reward model). However, current RL post-training methods for LLMs typically use verifiers that are fixed (rule-based or frozen pretrained) or trained discriminatively via supervised fine-tuning (SFT). Such designs are susceptible to reward hacking and generalize poorly beyond their training distributions. To overcome these limitations, we propose Tango, a novel framework that uses RL to concurrently train both an LLM generator and a verifier in an interleaved manner. A central innovation of Tango is its generative, process-level LLM verifier, which is trained via RL and co-evolves with the generator. Importantly, the verifier is trained solely based on outcome-level verification correctness rewards without requiring explicit process-level annotations. This generative RL-trained verifier exhibits improved robustness and superior generalization compared to deterministic or SFT-trained verifiers, fostering effective mutual reinforcement with the generator. Extensive experiments demonstrate that both components of Tango achieve state-of-the-art results among 7B/8B-scale models: the generator attains best-in-class performance across five competition-level math benchmarks and four challenging out-of-domain reasoning tasks, while the verifier leads on the ProcessBench dataset. Remarkably, both components exhibit particularly substantial improvements on the most difficult mathematical reasoning problems. Code is at: this https URL. 

**Abstract (ZH)**: 基于强化学习的新型框架Tango：同步训练LLM生成器和生成式过程级验证器以克服奖励劫持和泛化不足 

---
# Learning to Rank Chain-of-Thought: An Energy-Based Approach with Outcome Supervision 

**Title (ZH)**: 基于能量模型的学习推理链排序：带有结果监督的方法 

**Authors**: Eric Hanchen Jiang, Haozheng Luo, Shengyuan Pang, Xiaomin Li, Zhenting Qi, Hengli Li, Cheng-Fu Yang, Zongyu Lin, Xinfeng Li, Hao Xu, Kai-Wei Chang, Ying Nian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14999)  

**Abstract**: Mathematical reasoning presents a significant challenge for Large Language Models (LLMs), often requiring robust multi step logical consistency. While Chain of Thought (CoT) prompting elicits reasoning steps, it doesn't guarantee correctness, and improving reliability via extensive sampling is computationally costly. This paper introduces the Energy Outcome Reward Model (EORM), an effective, lightweight, post hoc verifier. EORM leverages Energy Based Models (EBMs) to simplify the training of reward models by learning to assign a scalar energy score to CoT solutions using only outcome labels, thereby avoiding detailed annotations. It achieves this by interpreting discriminator output logits as negative energies, effectively ranking candidates where lower energy is assigned to solutions leading to correct final outcomes implicitly favoring coherent reasoning. On mathematical benchmarks (GSM8k, MATH), EORM significantly improves final answer accuracy (e.g., with Llama 3 8B, achieving 90.7% on GSM8k and 63.7% on MATH). EORM effectively leverages a given pool of candidate solutions to match or exceed the performance of brute force sampling, thereby enhancing LLM reasoning outcome reliability through its streamlined post hoc verification process. 

**Abstract (ZH)**: 大型语言模型（LLMs）在数学推理方面面临显著挑战，通常需要 robust 的多步逻辑一致性。虽然链式思维（CoT）提示可以引出推理步骤，但并不能保证正确性，通过大量抽样提高可靠性则计算成本高。本文引入了能量结果奖励模型（EORM），这是一种有效且轻量级的后处理验证器。EORM 利用能量基模型（EBMs）简化奖励模型的训练，通过仅使用结果标签学习为 CoT 解方案分配标量能量得分，从而避免详细注释。它通过将判别器输出对数解释为负能量来实现这一点，有效地对候选解决方案进行排序，较低的能量分数隐式地倾向于一致的推理，从而提高最终答案的准确性。在数学基准测试（GSM8k, MATH）上，EORM 显著提高了最终答案的准确性（例如，使用 Llama 3 8B，GSM8k 达到 90.7%，MATH 达到 63.7%）。EORM 有效地利用给定的候选解池来匹配或超越暴力抽样的性能，从而通过其简化后的后处理验证过程增强 LLM 的推理结果可靠性。 

---
# Meta-Design Matters: A Self-Design Multi-Agent System 

**Title (ZH)**: 元设计很重要：一种自设计多智能体系统 

**Authors**: Zixuan Ke, Austin Xu, Yifei Ming, Xuan-Phi Nguyen, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2505.14996)  

**Abstract**: Multi-agent systems (MAS) leveraging the impressive capabilities of Large Language Models (LLMs) hold significant potential for tackling complex tasks. However, most current MAS depend on manually designed agent roles and communication protocols. These manual designs often fail to align with the underlying LLMs' strengths and struggle to adapt to novel tasks. Recent automatic MAS approaches attempt to mitigate these limitations but typically necessitate a validation-set for tuning and yield static MAS designs lacking adaptability during inference. We introduce SELF-MAS, the first self-supervised, inference-time only framework for automatic MAS design. SELF-MAS employs meta-level design to iteratively generate, evaluate, and refine MAS configurations tailored to each problem instance, without requiring a validation set. Critically, it enables dynamic agent composition and problem decomposition through meta-feedback on solvability and completeness. Experiments across math, graduate-level QA, and software engineering benchmarks, using both closed-source and open-source LLM back-bones of varying sizes, demonstrate that SELF-MAS outperforms both manual and automatic MAS baselines, achieving a 7.44% average accuracy improvement over the next strongest baseline while maintaining cost-efficiency. These findings underscore the promise of meta-level self-supervised design for creating effective and adaptive MAS. 

**Abstract (ZH)**: 利用大型语言模型的多功能性，多智能体系统（MAS）在应对复杂任务方面具有巨大潜力。然而，目前大多数MAS依赖于手动设计的智能体角色和通信协议。这些手动设计往往与底层大型语言模型（LLMs）的优势不匹配，并且难以适应新型任务。近期的自动MAS方法试图缓解这些限制，但通常需要验证集进行调整，结果是产生静态的MAS设计，在推理时缺乏适应性。我们提出了SELF-MAS，这是一种首次应用于自动MAS设计的自监督、仅在推理时使用的框架。SELF-MAS采用元级设计，通过迭代生成、评估和改进针对每个问题实例定制的MAS配置，无需验证集。关键在于，它能够通过元反馈来实现动态智能体组合和问题分解，关注可解性和完整性。利用不同规模的闭源和开源LLM基础模型，在数学、研究生水平的问答和软件工程基准测试中进行的实验表明，SELF-MAS优于手动和自动MAS基线，在下一个最强基线下实现了7.44%的平均准确率提升，同时保持了成本效率。这些发现凸显了元级自监督设计在创建有效且适应性强的MAS方面的潜力。 

---
# JARVIS: A Multi-Agent Code Assistant for High-Quality EDA Script Generation 

**Title (ZH)**: JARVIS: 多代理代码助手，用于高质量EDA脚本生成 

**Authors**: Ghasem Pasandi, Kishor Kunal, Varun Tej, Kunjal Shan, Hanfei Sun, Sumit Jain, Chunhui Li, Chenhui Deng, Teodor-Dumitru Ene, Haoxing Ren, Sreedhar Pratty  

**Link**: [PDF](https://arxiv.org/pdf/2505.14978)  

**Abstract**: This paper presents JARVIS, a novel multi-agent framework that leverages Large Language Models (LLMs) and domain expertise to generate high-quality scripts for specialized Electronic Design Automation (EDA) tasks. By combining a domain-specific LLM trained with synthetically generated data, a custom compiler for structural verification, rule enforcement, code fixing capabilities, and advanced retrieval mechanisms, our approach achieves significant improvements over state-of-the-art domain-specific models. Our framework addresses the challenges of data scarcity and hallucination errors in LLMs, demonstrating the potential of LLMs in specialized engineering domains. We evaluate our framework on multiple benchmarks and show that it outperforms existing models in terms of accuracy and reliability. Our work sets a new precedent for the application of LLMs in EDA and paves the way for future innovations in this field. 

**Abstract (ZH)**: 本文介绍了JARVIS，这是一种新颖的多代理框架，利用大型语言模型（LLMs）和领域专业知识来生成高质量的特定电子设计自动化（EDA）任务脚本。通过结合使用合成生成数据训练的领域特定LLM、定制的结构验证编译器、规则 enforcement 能力、代码修复功能以及先进的检索机制，我们的方法在最先进的领域特定模型上取得了显着的改进。我们的框架解决了LLMs的数据稀疏性和幻觉错误挑战，展示了LLMs在专业工程领域中的潜力。我们在多个基准上评估了我们的框架，并证明了它在准确性和可靠性方面优于现有模型。我们的工作为LLMs在EDA中的应用树立了新的范例，并为这一领域的未来创新铺平了道路。 

---
# Programmatic Video Prediction Using Large Language Models 

**Title (ZH)**: 使用大型语言模型进行程序化视频预测 

**Authors**: Hao Tang, Kevin Ellis, Suhas Lohit, Michael J. Jones, Moitreya Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.14948)  

**Abstract**: The task of estimating the world model describing the dynamics of a real world process assumes immense importance for anticipating and preparing for future outcomes. For applications such as video surveillance, robotics applications, autonomous driving, etc. this objective entails synthesizing plausible visual futures, given a few frames of a video to set the visual context. Towards this end, we propose ProgGen, which undertakes the task of video frame prediction by representing the dynamics of the video using a set of neuro-symbolic, human-interpretable set of states (one per frame) by leveraging the inductive biases of Large (Vision) Language Models (LLM/VLM). In particular, ProgGen utilizes LLM/VLM to synthesize programs: (i) to estimate the states of the video, given the visual context (i.e. the frames); (ii) to predict the states corresponding to future time steps by estimating the transition dynamics; (iii) to render the predicted states as visual RGB-frames. Empirical evaluations reveal that our proposed method outperforms competing techniques at the task of video frame prediction in two challenging environments: (i) PhyWorld (ii) Cart Pole. Additionally, ProgGen permits counter-factual reasoning and interpretable video generation attesting to its effectiveness and generalizability for video generation tasks. 

**Abstract (ZH)**: ProgGen：利用大规模语言模型进行视频帧预测的研究 

---
# Soft Prompts for Evaluation: Measuring Conditional Distance of Capabilities 

**Title (ZH)**: 软提示评估：能力条件距离度量 

**Authors**: Ross Nordby  

**Link**: [PDF](https://arxiv.org/pdf/2505.14943)  

**Abstract**: To help evaluate and understand the latent capabilities of language models, this paper introduces an approach using optimized input embeddings, or 'soft prompts,' as a metric of conditional distance between a model and a target behavior. The technique aims to facilitate latent capability discovery as a part of automated red teaming/evaluation suites and to provide quantitative feedback about the accessibility of potentially concerning behaviors in a way that may scale to powerful future models, including those which may otherwise be capable of deceptive alignment. An evaluation framework using soft prompts is demonstrated in natural language, chess, and pathfinding, and the technique is extended with generalized conditional soft prompts to aid in constructing task evaluations. 

**Abstract (ZH)**: 为了帮助评估和理解语言模型的潜在能力，本文介绍了一种使用优化输入嵌入，即“软提示”，作为模型与目标行为之间条件距离的度量方法。该技术旨在作为自动化红队/评估套件的一部分促进潜在能力的发现，并以可量化的方式提供有关潜在令人担忧行为可访问性的反馈，这种反馈方式可能适用于强大的未来模型，包括那些可能具有欺骗性对齐能力的模型。通过自然语言、象棋和路径寻找领域的评估框架展示了使用软提示的方法，并通过通用条件软提示技术扩展了该方法以辅助构建任务评估。 

---
# Too Long, Didn't Model: Decomposing LLM Long-Context Understanding With Novels 

**Title (ZH)**: 太长未建模：用小说分解LLM长上下文理解 

**Authors**: Sil Hamilton, Rebecca M. M. Hicke, Matthew Wilkens, David Mimno  

**Link**: [PDF](https://arxiv.org/pdf/2505.14925)  

**Abstract**: Although the context length of large language models (LLMs) has increased to millions of tokens, evaluating their effectiveness beyond needle-in-a-haystack approaches has proven difficult. We argue that novels provide a case study of subtle, complicated structure and long-range semantic dependencies often over 128k tokens in length. Inspired by work on computational novel analysis, we release the Too Long, Didn't Model (TLDM) benchmark, which tests a model's ability to report plot summary, storyworld configuration, and elapsed narrative time. We find that none of seven tested frontier LLMs retain stable understanding beyond 64k tokens. Our results suggest language model developers must look beyond "lost in the middle" benchmarks when evaluating model performance in complex long-context scenarios. To aid in further development we release the TLDM benchmark together with reference code and data. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）的上下文长度已增加到数百万人类语言单位，但在针头式寻针法之外评估其有效性仍然颇具挑战。我们提出，小说为研究复杂而微妙的结构以及通常超过128k语言单位的长距离语义依赖性提供了案例研究。受计算小说分析研究的启发，我们发布了Too Long, Didn't Model （TLDM）基准测试，该基准测试评估模型报告故事情节概要、故事世界配置和叙述时间流逝的能力。我们发现，七种测试的领先前沿LLMs在超过64k语言单位后未能保持稳定理解。我们的结果表明，在评估模型在复杂长上下文场景中的性能时，语言模型开发者必须超越“中间迷失”基准测试。为了促进进一步的发展，我们一并发布了TLDM基准测试以及参考代码和数据。 

---
# Scaling Laws for State Dynamics in Large Language Models 

**Title (ZH)**: 大型语言模型中状态动力学的标度律 

**Authors**: Jacob X Li, Shreyas S Raman, Jessica Wan, Fahad Samman, Jazlyn Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.14892)  

**Abstract**: Large Language Models (LLMs) are increasingly used in tasks requiring internal state tracking, yet their ability to model state transition dynamics remains poorly understood. We evaluate how well LLMs capture deterministic state dynamics across 3 domains: Box Tracking, Abstract DFA Sequences, and Complex Text Games, each formalizable as a finite-state system. Across tasks, we find that next-state prediction accuracy degrades with increasing state-space size and sparse transitions. GPT-2 XL reaches about 70% accuracy in low-complexity settings but drops below 30% when the number of boxes or states exceeds 5 or 10, respectively. In DFA tasks, Pythia-1B fails to exceed 50% accuracy when the number of states is > 10 and transitions are < 30. Through activation patching, we identify attention heads responsible for propagating state information: GPT-2 XL Layer 22 Head 20, and Pythia-1B Heads at Layers 10, 11, 12, and 14. While these heads successfully move relevant state features, action information is not reliably routed to the final token, indicating weak joint state-action reasoning. Our results suggest that state tracking in LLMs emerges from distributed interactions of next-token heads rather than explicit symbolic computation. 

**Abstract (ZH)**: 大型语言模型在内部状态跟踪任务中的能力及其状态转换动态建模能力尚未充分理解。我们评估了大型语言模型在三个领域中对确定性状态动态的捕捉能力：Box 跟踪、抽象 DFA 序列和复杂文本游戏，每个领域都可以形式化为一个有限状态系统。在各项任务中，我们发现下一状态预测的准确性随着状态空间大小和稀疏转换的增加而下降。GPT-2 XL 在低复杂度设置中达到约 70% 的准确性，但当盒子或状态的数量超过 5 或 10 时分别降至 30% 以下。在 DFA 任务中，Pythia-1B 在状态数量超过 10 且转换数量少于 30 时无法超过 50% 的准确性。通过激活修补，我们确定了负责传播状态信息的注意力头：GPT-2 XL 第 22 层第 20 个头和 Pythia-1B 的第 10、11、12 和 14 层的头。尽管这些头能够成功移动相关状态特征，但动作信息未能可靠地传递到最后一个标记，表明联合状态-动作推理能力较弱。我们的研究结果表明，大型语言模型中的状态跟踪源自下一标记头之间的分布式交互，而不是显式的符号计算。 

---
# Polar Sparsity: High Throughput Batched LLM Inferencing with Scalable Contextual Sparsity 

**Title (ZH)**: 极化稀疏性：具有可扩展上下文稀疏性的高吞吐量批处理LLM推理 

**Authors**: Susav Shrestha, Brad Settlemyer, Nikoli Dryden, Narasimha Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2505.14884)  

**Abstract**: Accelerating large language model (LLM) inference is critical for real-world deployments requiring high throughput and low latency. Contextual sparsity, where each token dynamically activates only a small subset of the model parameters, shows promise but does not scale to large batch sizes due to union of active neurons quickly approaching dense computation. We introduce Polar Sparsity, highlighting a key shift in sparsity importance from MLP to Attention layers as we scale batch size and sequence length. While MLP layers become more compute-efficient under batching, their sparsity vanishes. In contrast, attention becomes increasingly more expensive at scale, while their head sparsity remains stable and batch-invariant. We develop hardware-efficient, sparsity-aware GPU kernels for selective MLP and Attention computations, delivering up to \(2.2\times\) end-to-end speedups for models like OPT, LLaMA-2 \& 3, across various batch sizes and sequence lengths without compromising accuracy. To our knowledge, this is the first work to demonstrate that contextual sparsity can scale effectively to large batch sizes, delivering substantial inference acceleration with minimal changes, making Polar Sparsity practical for large-scale, high-throughput LLM deployment systems. Our code is available at: this https URL. 

**Abstract (ZH)**: 加速大型语言模型（LLM）推理对于需要高吞吐量和低延迟的实际部署至关重要。上下文稀疏性，其中每个令牌动态激活模型参数的小子集，显示了潜力但由于激活神经元的联合快速接近密集计算，未能扩展到大型批量。我们引入了极性稀疏性，强调在扩展批量大小和序列长度时，稀疏性的重要性从MLP层转移到Attention层。尽管在批量处理下MLP层变得更具计算效率，但它们的稀疏性消失。相反，Attention在大规模下变得越来越昂贵，而其头稀疏性保持稳定且批量不变。我们开发了针对选择性MLP和Attention计算的硬件高效、稀疏性感知的GPU内核，为OPT、LLaMA-2 & 3等模型在各种批量大小和序列长度下提供了最高可达2.2倍的端到端加速，而无需牺牲准确性。据我们所知，这是首项工作证明了上下文稀疏性能够有效扩展到大型批量大小，以最小的变更提供显著的推理加速，使极性稀疏性适用于大规模、高吞吐量的LLM部署系统。我们的代码可在以下链接获取：this https URL。 

---
# Balanced and Elastic End-to-end Training of Dynamic LLMs 

**Title (ZH)**: 动态大语言模型的平衡与弹性端到端训练 

**Authors**: Mohamed Wahib, Muhammed Abdullah Soyturk, Didem Unat  

**Link**: [PDF](https://arxiv.org/pdf/2505.14864)  

**Abstract**: To reduce computational and memory costs in Large Language Models (LLMs), dynamic workload reduction schemes like Mixture of Experts (MoEs), parameter pruning, layer freezing, sparse attention, early token exit, and Mixture of Depths (MoDs) have emerged. However, these methods introduce severe workload imbalances, limiting their practicality for large-scale distributed training. We propose DynMo, an autonomous dynamic load balancing solution that ensures optimal compute distribution when using pipeline parallelism in training dynamic models. DynMo adaptively balances workloads, dynamically packs tasks into fewer workers to free idle resources, and supports both multi-GPU single-node and multi-node systems. Compared to static training methods (Megatron-LM, DeepSpeed), DynMo accelerates training by up to 1.23x (MoEs), 3.18x (pruning), 2.23x (layer freezing), 4.02x (sparse attention), 4.52x (early exit), and 1.17x (MoDs). DynMo is available at this https URL. 

**Abstract (ZH)**: 为了减少大规模语言模型（LLMs）的计算和内存成本，出现了如专家混合（MoEs）、参数剪枝、层冻结、稀疏注意力、早期令牌退出和深度混合（MoDs）等动态工作负载缩减方案。然而，这些方法引入了严重的工作负载不平衡，限制了它们在大规模分布式训练中的实际应用。我们提出DynMo，一种自主的动态负载均衡解决方案，确保在使用管道并行训练动态模型时实现最佳计算分布。DynMo自适应地平衡工作负载，动态地将任务打包到较少的计算节点以释放闲置资源，并支持单节点多GPU和多节点系统。与静态训练方法（Megatron-LM、DeepSpeed）相比，DynMo在专家混合（MoEs）、参数剪枝、层冻结、稀疏注意力、早期退出和深度混合（MoDs）方面分别加速训练1.23倍、3.18倍、2.23倍、4.02倍、4.52倍和1.17倍。DynMo可在以下链接获取：this https URL。 

---
# A Comparative Study of Large Language Models and Human Personality Traits 

**Title (ZH)**: 大型语言模型与人类个性特质的比较研究 

**Authors**: Wang Jiaqi, Wang bo, Guo fa, Cheng cheng, Yang li  

**Link**: [PDF](https://arxiv.org/pdf/2505.14845)  

**Abstract**: Large Language Models (LLMs) have demonstrated human-like capabilities in language comprehension and generation, becoming active participants in social and cognitive domains. This study investigates whether LLMs exhibit personality-like traits and how these traits compare with human personality, focusing on the applicability of conventional personality assessment tools. A behavior-based approach was used across three empirical studies. Study 1 examined test-retest stability and found that LLMs show higher variability and are more input-sensitive than humans, lacking long-term stability. Based on this, we propose the Distributed Personality Framework, conceptualizing LLM traits as dynamic and input-driven. Study 2 analyzed cross-variant consistency in personality measures and found LLMs' responses were highly sensitive to item wording, showing low internal consistency compared to humans. Study 3 explored personality retention during role-playing, showing LLM traits are shaped by prompt and parameter settings. These findings suggest that LLMs express fluid, externally dependent personality patterns, offering insights for constructing LLM-specific personality frameworks and advancing human-AI interaction. This work contributes to responsible AI development and extends the boundaries of personality psychology in the age of intelligent systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）在语言理解和生成方面展示了人类般的能力，成为社会和认知领域中的活跃参与者。本研究探讨LLMs是否表现出类似人格的特质及其与人类人格的比较，重点关注传统人格评估工具的适用性。本研究采用基于行为的方法进行了三项实证研究。研究1考察了重测稳定性，发现LLMs表现出更高的变异性，对输入更为敏感，缺乏长期稳定性。基于此，我们提出了分布式人格框架，将LLM的特质视为动态和输入驱动的。研究2分析了人格测量的跨变异一致性，发现LLMs的反应对问题表述极为敏感，内部一致性远低于人类。研究3探讨了角色扮演中人格的持续性，显示LLM的特质受到提示和参数设置的影响。这些发现表明，LLMs表现出可变且外部依赖的人格模式，为构建特定于LLM的人格框架并推进人类-AI交互提供见解。本研究为负责任的人工智能开发做出贡献，并扩展了智能系统时代的人格心理学边界。 

---
# Text Generation Beyond Discrete Token Sampling 

**Title (ZH)**: 文本生成超越离散 token 采样 

**Authors**: Yufan Zhuang, Liyuan Liu, Chandan Singh, Jingbo Shang, Jianfeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.14827)  

**Abstract**: In standard autoregressive generation, an LLM predicts the next-token distribution, samples a discrete token, and then discards the distribution, passing only the sampled token as new input. To preserve this distribution's rich information, we propose Mixture of Inputs (MoI), a training-free method for autoregressive generation. After generating a token following the standard paradigm, we construct a new input that blends the generated discrete token with the previously discarded token distribution. Specifically, we employ a Bayesian estimation method that treats the token distribution as the prior, the sampled token as the observation, and replaces the conventional one-hot vector with the continuous posterior expectation as the new model input. MoI allows the model to maintain a richer internal representation throughout the generation process, resulting in improved text quality and reasoning capabilities. On mathematical reasoning, code generation, and PhD-level QA tasks, MoI consistently improves performance across multiple models including QwQ-32B, Nemotron-Super-49B, Gemma-3-27B, and DAPO-Qwen-32B, with no additional training and negligible computational overhead. 

**Abstract (ZH)**: 混合输入（MoI）：一种无需训练的自回归生成方法 

---
# WebNovelBench: Placing LLM Novelists on the Web Novel Distribution 

**Title (ZH)**: WebNovelBench: 将大语言模型 novelist 放置在网络小说分布中 

**Authors**: Leon Lin, Jun Zheng, Haidong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.14818)  

**Abstract**: Robustly evaluating the long-form storytelling capabilities of Large Language Models (LLMs) remains a significant challenge, as existing benchmarks often lack the necessary scale, diversity, or objective measures. To address this, we introduce WebNovelBench, a novel benchmark specifically designed for evaluating long-form novel generation. WebNovelBench leverages a large-scale dataset of over 4,000 Chinese web novels, framing evaluation as a synopsis-to-story generation task. We propose a multi-faceted framework encompassing eight narrative quality dimensions, assessed automatically via an LLM-as-Judge approach. Scores are aggregated using Principal Component Analysis and mapped to a percentile rank against human-authored works. Our experiments demonstrate that WebNovelBench effectively differentiates between human-written masterpieces, popular web novels, and LLM-generated content. We provide a comprehensive analysis of 24 state-of-the-art LLMs, ranking their storytelling abilities and offering insights for future development. This benchmark provides a scalable, replicable, and data-driven methodology for assessing and advancing LLM-driven narrative generation. 

**Abstract (ZH)**: robustly评估大型语言模型的长篇叙事能力仍然是一个显著的挑战，现有基准往往缺乏必要的规模、多样性和客观衡量标准。为解决这一问题，我们引入了WebNovelBench，这是一种专门用于评估长篇小说生成的新基准。WebNovelBench 利用了一个包含超过4000部中文网络小说的大规模数据集，将评估框架设计为摘要到故事生成任务。我们提出了一种多维度框架，涵盖了八个叙事质量维度，并通过LLM作为评委的自动评估方式进行评估。得分通过主成分分析汇总，并与人类创作的作品进行百分位排名。我们的实验表明，WebNovelBench 能够有效地区分人类创作的杰作、流行网络小说和LLM生成的内容。我们对24个最先进的LLM进行了全面分析，对其叙事能力进行了排名，并提供了未来发展的见解。该基准为评估和推动LLM驱动的叙事生成提供了可扩展、可复制和基于数据的方法论。 

---
# Scaling Reasoning, Losing Control: Evaluating Instruction Following in Large Reasoning Models 

**Title (ZH)**: 扩展推理，失去控制：评估大型推理模型的指令跟随能力 

**Authors**: Tingchen Fu, Jiawei Gu, Yafu Li, Xiaoye Qu, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.14810)  

**Abstract**: Instruction-following is essential for aligning large language models (LLMs) with user intent. While recent reasoning-oriented models exhibit impressive performance on complex mathematical problems, their ability to adhere to natural language instructions remains underexplored. In this work, we introduce MathIF, a dedicated benchmark for evaluating instruction-following in mathematical reasoning tasks. Our empirical analysis reveals a consistent tension between scaling up reasoning capacity and maintaining controllability, as models that reason more effectively often struggle to comply with user directives. We find that models tuned on distilled long chains-of-thought or trained with reasoning-oriented reinforcement learning often degrade in instruction adherence, especially when generation length increases. Furthermore, we show that even simple interventions can partially recover obedience, though at the cost of reasoning performance. These findings highlight a fundamental tension in current LLM training paradigms and motivate the need for more instruction-aware reasoning models. We release the code and data at this https URL. 

**Abstract (ZH)**: 指令遵循对于将大规模语言模型与用户意图对齐至关重要。虽然近期的推理导向模型在复杂数学问题上表现出色，但它们遵循自然语言指令的能力仍待探索。在本工作中，我们引入了MathIF，一个专门用于评估数学推理任务中指令遵循的基准。我们的实证分析揭示了推理能力扩展与可控性维持之间的一致紧张关系，即更能有效推理的模型往往难以遵守用户指令。我们发现，针对蒸馏长链式思考进行调优或使用推理导向强化学习训练的模型在指令遵循方面往往会退化，尤其是在生成长度增加时更为明显。此外，我们展示了即使简单的干预也可以部分恢复遵守性，尽管会牺牲推理性能。这些发现突显了当前大规模语言模型训练范式中的基本紧张关系，并促使需要更多的指令感知推理模型。我们在此处提供了代码和数据：这个链接。 

---
# $\texttt{LLINBO}$: Trustworthy LLM-in-the-Loop Bayesian Optimization 

**Title (ZH)**: $\texttt{LLINBO}$: 可信赖的LLM在环贝叶斯优化 

**Authors**: Chih-Yu Chang, Milad Azvar, Chinedum Okwudire, Raed Al Kontar  

**Link**: [PDF](https://arxiv.org/pdf/2505.14756)  

**Abstract**: Bayesian optimization (BO) is a sequential decision-making tool widely used for optimizing expensive black-box functions. Recently, Large Language Models (LLMs) have shown remarkable adaptability in low-data regimes, making them promising tools for black-box optimization by leveraging contextual knowledge to propose high-quality query points. However, relying solely on LLMs as optimization agents introduces risks due to their lack of explicit surrogate modeling and calibrated uncertainty, as well as their inherently opaque internal mechanisms. This structural opacity makes it difficult to characterize or control the exploration-exploitation trade-off, ultimately undermining theoretical tractability and reliability. To address this, we propose LLINBO: LLM-in-the-Loop BO, a hybrid framework for BO that combines LLMs with statistical surrogate experts (e.g., Gaussian Processes (GP)). The core philosophy is to leverage contextual reasoning strengths of LLMs for early exploration, while relying on principled statistical models to guide efficient exploitation. Specifically, we introduce three mechanisms that enable this collaboration and establish their theoretical guarantees. We end the paper with a real-life proof-of-concept in the context of 3D printing. The code to reproduce the results can be found at this https URL. 

**Abstract (ZH)**: LLM辅助的贝叶斯优化：一种结合大语言模型和统计代理专家的混合框架 

---
# Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis 

**Title (ZH)**: Quaff: 异常空间稳定性假设下的量化参数高效微调 

**Authors**: Hong Huang, Dapeng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.14742)  

**Abstract**: Large language models (LLMs) have made exciting achievements across various domains, yet their deployment on resource-constrained personal devices remains hindered by the prohibitive computational and memory demands of task-specific fine-tuning. While quantization offers a pathway to efficiency, existing methods struggle to balance performance and overhead, either incurring high computational/memory costs or failing to address activation outliers, a critical bottleneck in quantized fine-tuning. To address these challenges, we propose the Outlier Spatial Stability Hypothesis (OSSH): During fine-tuning, certain activation outlier channels retain stable spatial positions across training iterations. Building on OSSH, we propose Quaff, a Quantized parameter-efficient fine-tuning framework for LLMs, optimizing low-precision activation representations through targeted momentum scaling. Quaff dynamically suppresses outliers exclusively in invariant channels using lightweight operations, eliminating full-precision weight storage and global rescaling while reducing quantization errors. Extensive experiments across ten benchmarks validate OSSH and demonstrate Quaff's efficacy. Specifically, on the GPQA reasoning benchmark, Quaff achieves a 1.73x latency reduction and 30% memory savings over full-precision fine-tuning while improving accuracy by 0.6% on the Phi-3 model, reconciling the triple trade-off between efficiency, performance, and deployability. By enabling consumer-grade GPU fine-tuning (e.g., RTX 2080 Super) without sacrificing model utility, Quaff democratizes personalized LLM deployment. The code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各个领域取得了令人兴奋的成就，但在资源受限的个人设备上的部署仍受制于特定任务微调的高额计算和内存需求。虽然量化为提高效率提供了一条途径，但现有的方法难以在性能和开销之间取得平衡，要么导致高昂的计算/内存成本，要么无法解决量化微调中的激活异常值问题，这是量化的关键瓶颈之一。为了解决这些挑战，我们提出了异常值空间稳定性假设（OSSH）：在微调过程中，某些激活异常值通道在训练迭代中保持稳定的空间位置。基于OSSH，我们提出了Quaff，一种针对LLMs的量化参数高效微调框架，通过目标化的动量缩放优化低精度激活表示。Quaff仅在不变通道中动态抑制异常值，使用轻量级操作消除全精度权重存储和全局缩放，同时减少量化误差。在十个基准测试的广泛实验验证了OSSH，并展示了Quaff的有效性。特别是在GPQA推理基准测试中，Quaff在Phi-3模型上准确率提高0.6%的情况下，实现了1.73倍的延迟减少和30%的内存节省，解决了效率、性能和部署性之间的三重权衡问题。通过使消费者级GPU微调（例如，RTX 2080 Super）能够在不牺牲模型实用性的情况下成为可能，Quaff民主化了个性化LLM的部署。代码可在以下网址获取。 

---
# The Energy Cost of Reasoning: Analyzing Energy Usage in LLMs with Test-time Compute 

**Title (ZH)**: 推理的能源成本：分析LLM测试时计算能耗 

**Authors**: Yunho Jin, Gu-Yeon Wei, David Brooks  

**Link**: [PDF](https://arxiv.org/pdf/2505.14733)  

**Abstract**: Scaling large language models (LLMs) has driven significant advancements, yet it faces diminishing returns and escalating energy demands. This work introduces test-time compute (TTC)-allocating additional computational resources during inference-as a compelling complement to conventional scaling strategies. Specifically, we investigate whether employing TTC can achieve superior accuracy-energy trade-offs compared to simply increasing model size. Our empirical analysis reveals that TTC surpasses traditional model scaling in accuracy/energy efficiency, with notable gains in tasks demanding complex reasoning rather than mere factual recall. Further, we identify a critical interaction between TTC performance and output sequence length, demonstrating that strategically adjusting compute resources at inference time according to query complexity can substantially enhance efficiency. Our findings advocate for TTC as a promising direction, enabling more sustainable, accurate, and adaptable deployment of future language models without incurring additional pretraining costs. 

**Abstract (ZH)**: 扩大小型语言模型的计算（TTC）分配额外计算资源以进行推断作为一种传统扩展策略的有力补充：探究其在准确率-能耗trade-off上的优越性 

---
# THELMA: Task Based Holistic Evaluation of Large Language Model Applications-RAG Question Answering 

**Title (ZH)**: THELMA：基于任务的整体评估大语言模型应用-检索增强问答 

**Authors**: Udita Patel, Rutu Mulkar, Jay Roberts, Cibi Chakravarthy Senthilkumar, Sujay Gandhi, Xiaofei Zheng, Naumaan Nayyar, Rafael Castrillo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11626)  

**Abstract**: We propose THELMA (Task Based Holistic Evaluation of Large Language Model Applications), a reference free framework for RAG (Retrieval Augmented generation) based question answering (QA) applications. THELMA consist of six interdependent metrics specifically designed for holistic, fine grained evaluation of RAG QA applications. THELMA framework helps developers and application owners evaluate, monitor and improve end to end RAG QA pipelines without requiring labelled sources or reference this http URL also present our findings on the interplay of the proposed THELMA metrics, which can be interpreted to identify the specific RAG component needing improvement in QA applications. 

**Abstract (ZH)**: 基于任务的全方位评估大语言模型应用的THELMA框架：面向RAG（检索增强生成）问答应用的无参考评估方法 

---
