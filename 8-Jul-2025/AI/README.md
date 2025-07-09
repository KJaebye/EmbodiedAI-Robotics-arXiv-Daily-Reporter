# When Chain of Thought is Necessary, Language Models Struggle to Evade Monitors 

**Title (ZH)**: 当需要进行推理时，语言模型难以避开监控 

**Authors**: Scott Emmons, Erik Jenner, David K. Elson, Rif A. Saurous, Senthooran Rajamanoharan, Heng Chen, Irhum Shafkat, Rohin Shah  

**Link**: [PDF](https://arxiv.org/pdf/2507.05246)  

**Abstract**: While chain-of-thought (CoT) monitoring is an appealing AI safety defense, recent work on "unfaithfulness" has cast doubt on its reliability. These findings highlight an important failure mode, particularly when CoT acts as a post-hoc rationalization in applications like auditing for bias. However, for the distinct problem of runtime monitoring to prevent severe harm, we argue the key property is not faithfulness but monitorability. To this end, we introduce a conceptual framework distinguishing CoT-as-rationalization from CoT-as-computation. We expect that certain classes of severe harm will require complex, multi-step reasoning that necessitates CoT-as-computation. Replicating the experimental setups of prior work, we increase the difficulty of the bad behavior to enforce this necessity condition; this forces the model to expose its reasoning, making it monitorable. We then present methodology guidelines to stress-test CoT monitoring against deliberate evasion. Applying these guidelines, we find that models can learn to obscure their intentions, but only when given significant help, such as detailed human-written strategies or iterative optimization against the monitor. We conclude that, while not infallible, CoT monitoring offers a substantial layer of defense that requires active protection and continued stress-testing. 

**Abstract (ZH)**: 尽管链式思考（CoT）监控是AI安全防御的一种有吸引力的方法，但近期关于“不忠实性”的研究对其可靠性提出了质疑。这些发现突显出一个重要的失败模式，尤其是在CoT在偏见审计等应用中作为事后合理化工具时。然而，对于防止运行时严重危害的 distinct 问题，我们认为关键属性不是忠实性而是可监控性。为此，我们引入了一个概念性框架，区分CoT作为合理化与CoT作为计算之间的差异。我们预期，某些类别的严重危害需要复杂的多步推理，这需要CoT作为计算。通过增加前期研究中实验设置的难度，我们迫使模型暴露其推理过程，从而使其变得可监控。然后，我们提出了方法指南，以压力测试CoT监控的规避行为。应用这些指南，我们发现模型可以学会隐藏其意图，但仅当给予显著帮助时，如详细的人写策略或多次针对监控的优化。我们得出结论，虽然不是万无一失，但CoT监控提供了一种重要的防御层，需要积极保护并持续压力测试。 

---
# Modeling Latent Partner Strategies for Adaptive Zero-Shot Human-Agent Collaboration 

**Title (ZH)**: 建模潜在合作伙伴策略以实现自适应零样本人类-代理协作 

**Authors**: Benjamin Li, Shuyang Shi, Lucia Romero, Huao Li, Yaqi Xie, Woojun Kim, Stefanos Nikolaidis, Michael Lewis, Katia Sycara, Simon Stepputtis  

**Link**: [PDF](https://arxiv.org/pdf/2507.05244)  

**Abstract**: In collaborative tasks, being able to adapt to your teammates is a necessary requirement for success. When teammates are heterogeneous, such as in human-agent teams, agents need to be able to observe, recognize, and adapt to their human partners in real time. This becomes particularly challenging in tasks with time pressure and complex strategic spaces where the dynamics can change rapidly. In this work, we introduce TALENTS, a strategy-conditioned cooperator framework that learns to represent, categorize, and adapt to a range of partner strategies, enabling ad-hoc teamwork. Our approach utilizes a variational autoencoder to learn a latent strategy space from trajectory data. This latent space represents the underlying strategies that agents employ. Subsequently, the system identifies different types of strategy by clustering the data. Finally, a cooperator agent is trained to generate partners for each type of strategy, conditioned on these clusters. In order to adapt to previously unseen partners, we leverage a fixed-share regret minimization algorithm that infers and adjusts the estimated partner strategy dynamically. We assess our approach in a customized version of the Overcooked environment, posing a challenging cooperative cooking task that demands strong coordination across a wide range of possible strategies. Using an online user study, we show that our agent outperforms current baselines when working with unfamiliar human partners. 

**Abstract (ZH)**: 协作任务中，能够适应队友是成功的一个必要条件。当队友具有异质性，例如在人类-代理团队中，代理需要能够实时观察、识别并适应其人类伙伴。特别是在具有时间压力和复杂战略空间的任务中，动态变化尤为迅速，这给适应带来极大挑战。在此项工作中，我们引入了TALENTS框架，这是一种基于策略条件的合作者框架，旨在学习表示、分类和适应一系列合作伙伴策略，从而实现临时团队协作。该方法利用变分自编码器从轨迹数据中学习潜在策略空间。该潜在空间表示代理所采用的基本策略。随后，系统通过聚类数据识别不同类型的策略。最后，训练一个合作者代理，使其在特定聚类条件下生成相应类型的合作伙伴。为了适应之前未见过的合作伙伴，我们采用了固定份额遗憾最小化算法，该算法能够动态推断并调整估计的伙伴策略。我们在定制化的Overcooked环境中评估了我们的方法，该环境提出了一个具有挑战性的合作烹饪任务，要求在广泛的可能策略组合中实现强大的协调。通过在线用户研究，我们表明我们的代理在与不熟悉的_human_合作伙伴合作时性能优于现有基准。 

---
# SciMaster: Towards General-Purpose Scientific AI Agents, Part I. X-Master as Foundation: Can We Lead on Humanity's Last Exam? 

**Title (ZH)**: SciMaster: 通用于科学的人工智能代理探索，第一部分——X-Master作为基础：我们能否通过人类的最后一考？ 

**Authors**: Jingyi Chai, Shuo Tang, Rui Ye, Yuwen Du, Xinyu Zhu, Mengcheng Zhou, Yanfeng Wang, Weinan E, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.05241)  

**Abstract**: The rapid advancements of AI agents have ignited the long-held ambition of leveraging them to accelerate scientific discovery. Achieving this goal requires a deep understanding of the frontiers of human knowledge. As such, Humanity's Last Exam (HLE) provides an exceptionally challenging touchstone for evaluating scientific AI agents. In this work, we aim to construct the foundational architecture for general-purpose agents and validate the capabilities through leading performance on HLE. To achieve this, we introduce X-Master, a tool-augmented reasoning agent designed to emulate human researchers by interacting flexibly with external tools during its reasoning process. This agent, guided by the conceptualization of code as an interaction language, can flexibly leverage built-in Python libraries and our customized tools to augment the reasoning. We further scale its capabilities through X-Masters, a scattered-and-stacked agentic workflow that systematically enhances breadth and depth of reasoning. Our open-source solution, X-Masters, sets a new state-of-the-art record on HLE with a score of 32.1%, surpassing OpenAI's and Google's Deep Research (26.6% and 26.9%) and becoming the first to exceed the 30% threshold. This work allows us to gain a deeper understanding of complex task-solving and accumulates valuable experience that can inform future advancements, guiding subsequent model training. 

**Abstract (ZH)**: AI代理的快速发展掀起了利用它们加速科学发现的长期愿景。实现这一目标需要深刻理解人类知识的前沿。为此，人类最后考试（HLE）提供了评估科学AI代理的极其具有挑战性的标准。在本文中，我们旨在构建通用代理的基础架构，并通过在HLE上的领先性能来验证其能力。为此，我们介绍了X-Master，一种工具增强的推理代理，旨在通过在其推理过程中灵活与外部工具交互来模仿人类研究人员。该代理根据代码作为交互语言的概念，可以灵活地利用内置的Python库和我们定制的工具来增强推理能力。我们进一步通过X-Masters的分散和堆叠代理工作流扩展其能力，该工作流系统地增强了推理的广度和深度。我们的开源解决方案X-Masters在HLE上取得了新的最佳成绩，得分为32.1%，超越了OpenAI的和Google的Deep Research（分别为26.6%和26.9%），成为首个超过30%阈值的系统。这项工作使我们能够更深入地了解复杂任务解决问题，并积累了宝贵的经验，这些经验可以指导未来的进步，指导后续模型训练。 

---
# MedGemma Technical Report 

**Title (ZH)**: MedGemma技术报告 

**Authors**: Andrew Sellergren, Sahar Kazemzadeh, Tiam Jaroensri, Atilla Kiraly, Madeleine Traverse, Timo Kohlberger, Shawn Xu, Fayaz Jamil, Cían Hughes, Charles Lau, Justin Chen, Fereshteh Mahvar, Liron Yatziv, Tiffany Chen, Bram Sterling, Stefanie Anna Baby, Susanna Maria Baby, Jeremy Lai, Samuel Schmidgall, Lu Yang, Kejia Chen, Per Bjornsson, Shashir Reddy, Ryan Brush, Kenneth Philbrick, Howard Hu, Howard Yang, Richa Tiwari, Sunny Jansen, Preeti Singh, Yun Liu, Shekoofeh Azizi, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Riviere, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean-bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Elena Buchatskaya, Jean-Baptiste Alayrac, Dmitry, Lepikhin, Vlad Feinberg, Sebastian Borgeaud, Alek Andreev, Cassidy Hardin, Robert Dadashi, Léonard Hussenot, Armand Joulin, Olivier Bachem, Yossi Matias, Katherine Chou, Avinatan Hassidim, Kavi Goel, Clement Farabet, Joelle Barral, Tris Warkentin, Jonathon Shlens, David Fleet, Victor Cotruta, Omar Sanseviero, Gus Martins, Phoebe Kirk, Anand Rao, Shravya Shetty, David F. Steiner, Can Kirmizibayrak, Rory Pilgrim, Daniel Golden, Lin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05201)  

**Abstract**: Artificial intelligence (AI) has significant potential in healthcare applications, but its training and deployment faces challenges due to healthcare's diverse data, complex tasks, and the need to preserve privacy. Foundation models that perform well on medical tasks and require less task-specific tuning data are critical to accelerate the development of healthcare AI applications. We introduce MedGemma, a collection of medical vision-language foundation models based on Gemma 3 4B and 27B. MedGemma demonstrates advanced medical understanding and reasoning on images and text, significantly exceeding the performance of similar-sized generative models and approaching the performance of task-specific models, while maintaining the general capabilities of the Gemma 3 base models. For out-of-distribution tasks, MedGemma achieves 2.6-10% improvement on medical multimodal question answering, 15.5-18.1% improvement on chest X-ray finding classification, and 10.8% improvement on agentic evaluations compared to the base models. Fine-tuning MedGemma further improves performance in subdomains, reducing errors in electronic health record information retrieval by 50% and reaching comparable performance to existing specialized state-of-the-art methods for pneumothorax classification and histopathology patch classification. We additionally introduce MedSigLIP, a medically-tuned vision encoder derived from SigLIP. MedSigLIP powers the visual understanding capabilities of MedGemma and as an encoder achieves comparable or better performance than specialized medical image encoders. Taken together, the MedGemma collection provides a strong foundation of medical image and text capabilities, with potential to significantly accelerate medical research and development of downstream applications. The MedGemma collection, including tutorials and model weights, can be found at this https URL. 

**Abstract (ZH)**: 人工智能（AI）在医疗应用中有巨大的潜力，但由于医疗数据的多样性、任务的复杂性以及需要保护隐私，其训练和部署面临着挑战。能够很好地完成医疗任务且需要较少的特定任务调优数据的基础模型对于加速医疗AI应用的发展至关重要。我们介绍了MedGemma，这是一种基于Gemma 3 4B和27B的医疗视觉-语言基础模型集合。MedGemma在图像和文本上展示了高级的医学理解和推理能力，显著超过了同类生成模型的性能，并接近特定任务模型的性能，同时保持了Gemma 3基础模型的一般能力。对于分布外任务，MedGemma在医疗多模态问答上的表现提高了2.6-10%，在胸部X光检查分类上的表现提高了15.5-18.1%，在代理评估上的表现提高了10.8%，优于基础模型。进一步微调MedGemma在子领域进一步提高了性能，减少了电子健康记录信息检索的错误50%，并在气胸分类和组织病理学斑块分类方面达到了现有专门方法的相当性能。此外，我们还介绍了MedSigLIP，这是一种基于SigLIP的医学调优视觉编码器。MedSigLIP增强了MedGemma的视觉理解能力，作为编码器，其性能与专门的医学图像编码器相当或更好。总体而言，MedGemma集合提供了一种强大的医学图像和文本能力基础，有望显著加速医学研究和下游应用的发展。MedGemma集合，包括教程和模型权重，可在以下链接找到：this https URL。 

---
# GIST: Cross-Domain Click-Through Rate Prediction via Guided Content-Behavior Distillation 

**Title (ZH)**: GIST：通过指导内容-行为提炼进行跨域点击率预测 

**Authors**: Wei Xu, Haoran Li, Baoyuan Ou, Lai Xu, Yingjie Qin, Ruilong Su, Ruiwen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.05142)  

**Abstract**: Cross-domain Click-Through Rate prediction aims to tackle the data sparsity and the cold start problems in online advertising systems by transferring knowledge from source domains to a target domain. Most existing methods rely on overlapping users to facilitate this transfer, often focusing on joint training or pre-training with fine-tuning approach to connect the source and target domains. However, in real-world industrial settings, joint training struggles to learn optimal representations with different distributions, and pre-training with fine-tuning is not well-suited for continuously integrating new data. To address these issues, we propose GIST, a cross-domain lifelong sequence model that decouples the training processes of the source and target domains. Unlike previous methods that search lifelong sequences in the source domains using only content or behavior signals or their simple combinations, we innovatively introduce a Content-Behavior Joint Training Module (CBJT), which aligns content-behavior distributions and combines them with guided information to facilitate a more stable representation. Furthermore, we develop an Asymmetric Similarity Integration strategy (ASI) to augment knowledge transfer through similarity computation. Extensive experiments demonstrate the effectiveness of GIST, surpassing SOTA methods on offline evaluations and an online A/B test. Deployed on the Xiaohongshu (RedNote) platform, GIST effectively enhances online ads system performance at scale, serving hundreds of millions of daily active users. 

**Abstract (ZH)**: 跨领域点击率预测旨在通过从源领域转移到目标领域来解决在线广告系统中的数据稀疏性和冷启动问题。大多数现有方法依赖于重叠用户来促进这一转移，通常集中于通过联合训练或预训练加微调的方法来连接源和目标领域。然而，在实际工业环境中，联合训练难以学习具有不同分布的最佳表示，而预训练加微调也不适合持续集成新数据。为了解决这些问题，我们提出了一种跨领域终身序列模型GIST，将源领域和目标领域的训练过程分离。与之前方法仅使用内容或行为信号及其简单组合在源领域中搜索终身序列不同，我们创新性地引入了内容-行为联合训练模块（CBJT），该模块对齐内容-行为分布并结合引导信息，以促进更稳定的表现。此外，我们开发了非对称相似性集成策略（ASI）以通过相似性计算增强知识转移。广泛的实验表明，GIST在离线评估和在线A/B测试中均优于当前最佳方法。部署在小红书（RedNote）平台上，GIST有效增强了大规模在线广告系统的性能，服务于数亿日活跃用户。 

---
# Rule Learning for Knowledge Graph Reasoning under Agnostic Distribution Shift 

**Title (ZH)**: agnostic分布迁移下的知识图谱推理规则学习 

**Authors**: Shixuan Liu, Yue He, Yunfei Wang, Hao Zou, Haoxiang Cheng, Wenjing Yang, Peng Cui, Zhong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.05110)  

**Abstract**: Knowledge graph (KG) reasoning remains a critical research area focused on inferring missing knowledge by analyzing relationships among observed facts. Despite its success, a key limitation of existing KG reasoning methods is their dependence on the I.I.D assumption. This assumption can easily be violated due to unknown sample selection bias during training or agnostic distribution shifts during testing, significantly compromising model performance and reliability. To facilitate the deployment of KG reasoning in wild environments, this study investigates learning logical rules from KGs affected by unknown selection bias. Additionally, we address test sets with agnostic distribution shifts, formally defining this challenge as out-of-distribution (OOD) KG reasoning-a previously underexplored problem. To solve the issue, we propose the Stable Rule Learning (StableRule) framework, an end-to-end methodology that integrates feature decorrelation with rule learning network, to enhance OOD generalization performance. By leveraging feature decorrelation, the StableRule framework mitigates the adverse effects of covariate shifts arising in OOD scenarios, thereby improving the robustness of the rule learning component in effectively deriving logical rules. Extensive experiments on seven benchmark KGs demonstrate the framework's superior effectiveness and stability across diverse heterogeneous environments, underscoring its practical significance for real-world applications. 

**Abstract (ZH)**: 知识图谱推理中的未知选择偏差下稳定逻辑规则学习 

---
# How Rules Represent Causal Knowledge: Causal Modeling with Abductive Logic Programs 

**Title (ZH)**: 规则如何表示因果知识：基于 abduction 逻辑程序的因果建模 

**Authors**: Kilian Rückschloß, Felix Weitkämper  

**Link**: [PDF](https://arxiv.org/pdf/2507.05088)  

**Abstract**: Pearl observes that causal knowledge enables predicting the effects of interventions, such as actions, whereas descriptive knowledge only permits drawing conclusions from observation. This paper extends Pearl's approach to causality and interventions to the setting of stratified abductive logic programs. It shows how stable models of such programs can be given a causal interpretation by building on philosophical foundations and recent work by Bochman and Eelink et al. In particular, it provides a translation of abductive logic programs into causal systems, thereby clarifying the informal causal reading of logic program rules and supporting principled reasoning about external actions. The main result establishes that the stable model semantics for stratified programs conforms to key philosophical principles of causation, such as causal sufficiency, natural necessity, and irrelevance of unobserved effects. This justifies the use of stratified abductive logic programs as a framework for causal modeling and for predicting the effects of interventions 

**Abstract (ZH)**: Pearl指出，因果知识能使我们预测干预（如行动）的效果，而描述性知识仅能从观察中得出结论。本文将Pearl的因果推理方法扩展到分层 abduction 逻辑程序的设置中。通过建立在哲学基础之上，并结合Bochman及Eelink等人近期的工作，展示了如何给这样的程序的稳定模型赋予因果解释。特别地，本文提供了将 abduction 逻辑程序翻译成因果系统的翻译方法，从而澄清了逻辑程序规则的非正式因果阅读，并支持对外部行动进行原则性的推理。主要结果表明，分层程序的稳定模型语义符合因果哲学原则的关键标准，如因果完备性、自然必要性和未观察效应的相关性不重要。这证明了使用分层 abduction 逻辑程序作为因果建模框架以及预测干预效果的有效性。 

---
# When Imitation Learning Outperforms Reinforcement Learning in Surgical Action Planning 

**Title (ZH)**: 当模仿学习在手术动作规划中表现优于强化学习时 

**Authors**: Maxence Boels, Harry Robertshaw, Alejandro Granados, Prokar Dasgupta, Sebastien Ourselin  

**Link**: [PDF](https://arxiv.org/pdf/2507.05011)  

**Abstract**: Surgical action planning requires predicting future instrument-verb-target triplets for real-time assistance. While teleoperated robotic surgery provides natural expert demonstrations for imitation learning (IL), reinforcement learning (RL) could potentially discover superior strategies through exploration. We present the first comprehensive comparison of IL versus RL for surgical action planning on CholecT50. Our Dual-task Autoregressive Imitation Learning (DARIL) baseline achieves 34.6% action triplet recognition mAP and 33.6% next frame prediction mAP with smooth planning degradation to 29.2% at 10-second horizons. We evaluated three RL variants: world model-based RL, direct video RL, and inverse RL enhancement. Surprisingly, all RL approaches underperformed DARIL i.e. world model RL dropped to 3.1% mAP at 10s while direct video RL achieved only 15.9%. Our analysis reveals that distribution matching on expert-annotated test sets systematically favors IL over potentially valid RL policies that differ from training demonstrations. This challenges assumptions about RL superiority in sequential decision making and provides crucial insights for surgical AI development. 

**Abstract (ZH)**: 手术动作规划需要预测未来的器械-动词-目标三元组以实现实时辅助。虽然远程操作的机器人手术提供了自然的 expert 示范用于模仿学习（IL），强化学习（RL）则有可能通过探索发现更优策略。我们首次在 CholecT50 上全面比较了 IL 与 RL 在手术动作规划中的应用。我们的双任务自回归模仿学习（DARIL）基线实现 34.6% 的动作三元组识别 mAP 和 33.6% 的下一帧预测 mAP，并且在 10 秒时间窗口内平滑下降到 29.2%。我们评估了三种 RL 变体：基于世界模型的 RL、直接视频 RL 和逆 RL 增强。令人惊讶的是，所有 RL 方法均劣于 DARIL，即基于世界模型的 RL 在 10 秒时仅达到 3.1% 的 mAP，而直接视频 RL 仅达到 15.9%。我们的分析表明，针对专家标注的测试集进行分布匹配系统性地偏向于 IL 而非差异训练示例的潜在有效 RL 策略。这挑战了在顺序决策中 RL 优越性的假设，并为手术 AI 的发展提供了重要见解。 

---
# Supported Abstract Argumentation for Case-Based Reasoning 

**Title (ZH)**: 基于支持的抽象论辩案例基于推理 

**Authors**: Adam Gould, Gabriel de Olim Gaul, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2507.04994)  

**Abstract**: We introduce Supported Abstract Argumentation for Case-Based Reasoning (sAA-CBR), a binary classification model in which past cases engage in debates by arguing in favour of their labelling and attacking or supporting those with opposing or agreeing labels. With supports, sAA-CBR overcomes the limitation of its precursor AA-CBR, which can contain extraneous cases (or spikes) that are not included in the debates. We prove that sAA-CBR contains no spikes, without trading off key model properties 

**Abstract (ZH)**: 基于支持的抽象论辩案例推理（sAA-CBR）：一种二分类模型 

---
# MARBLE: A Multi-Agent Rule-Based LLM Reasoning Engine for Accident Severity Prediction 

**Title (ZH)**: MARBLE：基于多Agent规则的LLM事故严重性预测推理引擎 

**Authors**: Kaleem Ullah Qasim, Jiashu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04893)  

**Abstract**: Accident severity prediction plays a critical role in transportation safety systems but is a persistently difficult task due to incomplete data, strong feature dependencies, and severe class imbalance in which rare but high-severity cases are underrepresented and hard to detect. Existing methods often rely on monolithic models or black box prompting, which struggle to scale in noisy, real-world settings and offer limited interpretability. To address these challenges, we propose MARBLE a multiagent rule based LLM engine that decomposes the severity prediction task across a team of specialized reasoning agents, including an interchangeable ML-backed agent. Each agent focuses on a semantic subset of features (e.g., spatial, environmental, temporal), enabling scoped reasoning and modular prompting without the risk of prompt saturation. Predictions are coordinated through either rule-based or LLM-guided consensus mechanisms that account for class rarity and confidence dynamics. The system retains structured traces of agent-level reasoning and coordination outcomes, supporting in-depth interpretability and post-hoc performance diagnostics. Across both UK and US datasets, MARBLE consistently outperforms traditional machine learning classifiers and state-of-the-art (SOTA) prompt-based reasoning methods including Chain-of-Thought (CoT), Least-to-Most (L2M), and Tree-of-Thought (ToT) achieving nearly 90% accuracy where others plateau below 48%. This performance redefines the practical ceiling for accident severity classification under real world noise and extreme class imbalance. Our results position MARBLE as a generalizable and interpretable framework for reasoning under uncertainty in safety-critical applications. 

**Abstract (ZH)**: 事故严重程度预测在交通安全系统中发挥着关键作用，但由于数据不完整、特征依赖性强以及严重类别不平衡（罕见但严重程度高的案例代表性不足且难以检测）等原因，这是一个持续困难的任务。现有方法通常依赖于单体模型或黑盒提示，难以在嘈杂的现实环境中扩展，并且缺乏解释性。为了解决这些问题，我们提出了一种名为MARBLE的多智能体基于规则的LLM引擎，该引擎将严重程度预测任务分解为一组专门推理智能体，包括一个可互换的机器学习支持智能体。每个智能体专注于语义子特征集（如空间、环境、时间），从而实现聚焦推理和模块化提示，避免提示饱和的风险。预测通过基于规则或LLM引导的共识机制协调，这些机制考虑了类别稀有性和信心动态。系统保留了智能体级推理和协调结果的结构化记录，支持深入解释和事后性能诊断。在英国和美国数据集上，MARBLE一致地优于传统机器学习分类器和最先进的基于提示的推理方法，包括思维链（CoT）、从小到大（L2M）和思维树（ToT），实现近90%的准确率，而其他方法在48%以下停滞不前。这一性能重新定义了在现实世界噪声和极端类别不平衡下事故严重程度分类的实际天花板。我们的结果将MARBLE定位为在安全关键应用中处理不确定性推理的可泛化和可解释框架。 

---
# DoPI: Doctor-like Proactive Interrogation LLM for Traditional Chinese Medicine 

**Title (ZH)**: DoPI: 医师般主动问询的大语言模型在中医中的应用 

**Authors**: Zewen Sun, Ruoxiang Huang, Jiahe Feng, Rundong Kong, Yuqian Wang, Hengyu Liu, Ziqi Gong, Yuyuan Qin, Yingxue Wang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04877)  

**Abstract**: Enhancing interrogation capabilities in Traditional Chinese Medicine (TCM) diagnosis through multi-turn dialogues and knowledge graphs presents a significant challenge for modern AI systems. Current large language models (LLMs), despite their advancements, exhibit notable limitations in medical applications, particularly in conducting effective multi-turn dialogues and proactive questioning. These shortcomings hinder their practical application and effectiveness in simulating real-world diagnostic scenarios. To address these limitations, we propose DoPI, a novel LLM system specifically designed for the TCM domain. The DoPI system introduces a collaborative architecture comprising a guidance model and an expert model. The guidance model conducts multi-turn dialogues with patients and dynamically generates questions based on a knowledge graph to efficiently extract critical symptom information. Simultaneously, the expert model leverages deep TCM expertise to provide final diagnoses and treatment plans. Furthermore, this study constructs a multi-turn doctor-patient dialogue dataset to simulate realistic consultation scenarios and proposes a novel evaluation methodology that does not rely on manually collected real-world consultation data. Experimental results show that the DoPI system achieves an accuracy rate of 84.68 percent in interrogation outcomes, significantly enhancing the model's communication ability during diagnosis while maintaining professional expertise. 

**Abstract (ZH)**: 通过多轮对话和知识图谱增强中医诊断问询能力：现代AI系统的挑战及DoPI系统的提出 

---
# Application and Evaluation of Large Language Models for Forecasting the Impact of Traffic Incidents 

**Title (ZH)**: 大型语言模型在预测交通事件影响中的应用与评估 

**Authors**: George Jagadeesh, Srikrishna Iyer, Michal Polanowski, Kai Xin Thia  

**Link**: [PDF](https://arxiv.org/pdf/2507.04803)  

**Abstract**: This study examines the feasibility of applying large language models (LLMs) for forecasting the impact of traffic incidents on the traffic flow. The use of LLMs for this task has several advantages over existing machine learning-based solutions such as not requiring a large training dataset and the ability to utilize free-text incident logs. We propose a fully LLM-based solution that predicts the incident impact using a combination of traffic features and LLM-extracted incident features. A key ingredient of this solution is an effective method of selecting examples for the LLM's in-context learning. We evaluate the performance of three advanced LLMs and two state-of-the-art machine learning models on a real traffic incident dataset. The results show that the best-performing LLM matches the accuracy of the most accurate machine learning model, despite the former not having been trained on this prediction task. The findings indicate that LLMs are a practically viable option for traffic incident impact prediction. 

**Abstract (ZH)**: 本研究探讨了使用大型语言模型（LLMs）预测交通 incident 对交通流量影响可行性的研究。该研究提出了一种基于LLM的解决方案，通过结合交通特征和LLM提取的incident特征来预测incident的影响。该解决方案的关键要素是为LLM的上下文学习有效选择示例的方法。研究在实际交通incident数据集上评估了三种高性能LLM和两种最先进的机器学习模型的性能。结果表明，表现最佳的LLM在准确度上与最精确的机器学习模型相当，尽管前者未针对此预测任务进行训练。研究发现表明LLM是交通incident影响预测的一种实际可行的选择。 

---
# FurniMAS: Language-Guided Furniture Decoration using Multi-Agent System 

**Title (ZH)**: FurniMAS：基于多agent系统的语言引导家具装饰 

**Authors**: Toan Nguyen, Tri Le, Quang Nguyen, Anh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.04770)  

**Abstract**: Furniture decoration is an important task in various industrial applications. However, achieving a high-quality decorative result is often time-consuming and requires specialized artistic expertise. To tackle these challenges, we explore how multi-agent systems can assist in automating the decoration process. We propose FurniMAS, a multi-agent system for automatic furniture decoration. Specifically, given a human prompt and a household furniture item such as a working desk or a TV stand, our system suggests relevant assets with appropriate styles and materials, and arranges them on the item, ensuring the decorative result meets functionality, aesthetic, and ambiance preferences. FurniMAS assembles a hybrid team of LLM-based and non-LLM agents, each fulfilling distinct roles in a typical decoration project. These agents collaborate through communication, logical reasoning, and validation to transform the requirements into the final outcome. Extensive experiments demonstrate that our FurniMAS significantly outperforms other baselines in generating high-quality 3D decor. 

**Abstract (ZH)**: 家具装饰是各种工业应用中的一个重要任务。然而，实现高质量的装饰效果往往耗时且需要专门的艺术技能。为应对这些挑战，我们探讨了多智能体系统如何协助自动化装饰过程。我们提出FurniMAS，一种用于自动家具装饰的多智能体系统。具体来说，给定人类提示和如办公桌或电视柜等家居家具项，我们的系统建议合适的样式和材料相关的资产，并将它们布置在家具项上，以确保装饰效果满足功能、美学和氛围偏好。FurniMAS 组建了一个基于LLM和非LLM智能体的混合团队，每个智能体在典型的装饰项目中承担不同的角色。这些智能体通过沟通、逻辑推理和验证合作，将需求转化为最终结果。大量实验表明，我们的FurniMAS在生成高质量3D装饰方面显著优于其他基准方法。 

---
# LLM-based Question-Answer Framework for Sensor-driven HVAC System Interaction 

**Title (ZH)**: 基于LLM的由传感器驱动的HVAC系统交互问答框架 

**Authors**: Sungmin Lee, Minju Kang, Joonhee Lee, Seungyong Lee, Dongju Kim, Jingi Hong, Jun Shin, Pei Zhang, JeongGil Ko  

**Link**: [PDF](https://arxiv.org/pdf/2507.04748)  

**Abstract**: Question-answering (QA) interfaces powered by large language models (LLMs) present a promising direction for improving interactivity with HVAC system insights, particularly for non-expert users. However, enabling accurate, real-time, and context-aware interactions with HVAC systems introduces unique challenges, including the integration of frequently updated sensor data, domain-specific knowledge grounding, and coherent multi-stage reasoning. In this paper, we present JARVIS, a two-stage LLM-based QA framework tailored for sensor data-driven HVAC system interaction. JARVIS employs an Expert-LLM to translate high-level user queries into structured execution instructions, and an Agent that performs SQL-based data retrieval, statistical processing, and final response generation. To address HVAC-specific challenges, JARVIS integrates (1) an adaptive context injection strategy for efficient HVAC and deployment-specific information integration, (2) a parameterized SQL builder and executor to improve data access reliability, and (3) a bottom-up planning scheme to ensure consistency across multi-stage response generation. We evaluate JARVIS using real-world data collected from a commercial HVAC system and a ground truth QA dataset curated by HVAC experts to demonstrate its effectiveness in delivering accurate and interpretable responses across diverse queries. Results show that JARVIS consistently outperforms baseline and ablation variants in both automated and user-centered assessments, achieving high response quality and accuracy. 

**Abstract (ZH)**: 由大型语言模型驱动的问答接口(JARVIS)：面向传感器数据驱动的暖通空调系统交互的两阶段框架 

---
# Activation Steering for Chain-of-Thought Compression 

**Title (ZH)**: 思维链压缩的激活方向控制 

**Authors**: Seyedarmin Azizi, Erfan Baghaei Potraghloo, Massoud Pedram  

**Link**: [PDF](https://arxiv.org/pdf/2507.04742)  

**Abstract**: Large language models (LLMs) excel at complex reasoning when they include intermediate steps, known as "chains of thought" (CoTs). However, these rationales are often overly verbose, even for simple problems, leading to wasted context, increased latency, and higher energy consumption. We observe that verbose, English-heavy CoTs and concise, math-centric CoTs occupy distinct regions in the model's residual-stream activation space. By extracting and injecting a "steering vector" to transition between these modes, we can reliably shift generation toward more concise reasoning, effectively compressing CoTs without retraining. We formalize this approach as Activation-Steered Compression (ASC), an inference-time technique that shortens reasoning traces by directly modifying hidden representations. In addition, we provide a theoretical analysis of the impact of ASC on the output distribution, derived from a closed-form KL-divergence-bounded constraint to regulate steering strength. Using only 100 paired verbose and concise examples, ASC achieves up to 67.43% reduction in CoT length on MATH500 and GSM8K datasets, while maintaining accuracy across 7B, 8B, and 32B parameter models. As a training-free method, ASC introduces negligible runtime overhead and, on MATH500, delivers an average 2.73x speedup in end-to-end reasoning wall-clock time on an 8B model. This makes ASC a practical and efficient tool for streamlining the deployment of reasoning-capable LLMs in latency- or cost-sensitive settings. The code is available at: this https URL 

**Abstract (ZH)**: 大型语言模型在包含中间步骤的“链式思考”（CoTs）的情况下能够进行复杂的推理，但在许多情况下，这些推理过程过于冗长，即使是对于简单问题也是如此，导致上下文浪费、延迟增加和能耗上升。我们发现，冗长的英语为主的CoTs和简洁的数学为中心的CoTs在模型的残差流激活空间中占据不同的区域。通过抽取和注入“控制向量”以在这些模式之间进行转换，可以可靠地将生成导向更简洁的推理，有效地压缩CoTs而无需重新训练。我们将这种方法形式化为激活控制压缩（ASC），这是一种推理时技术，通过直接修改隐藏表示来缩短推理轨迹。此外，我们还提供了一种针对ASC输出分布的影响的理论分析，通过闭式KL散度约束来调节控制强度。仅使用100对冗长和简洁的例子，ASC在MATH500和GSM8K数据集上实现了高达67.43%的CoT长度减少，同时在7B、8B和32B参数模型中保持了准确性。作为一种无需训练的方法，ASC引入了微乎其微的运行时开销，并在MATH500上实现了8B模型端到端推理时间平均2.73倍的加速。这使得ASC成为在延迟或成本敏感环境中 streamlined 部署具备推理能力的LLMs的一种实用和高效工具。代码可在以下链接获取：this https URL。 

---
# ChipSeek-R1: Generating Human-Surpassing RTL with LLM via Hierarchical Reward-Driven Reinforcement Learning 

**Title (ZH)**: ChipSeek-R1: 通过层次奖励驱动强化学习生成 surpass 人类的RTL代码 

**Authors**: Zhirong Chen, Kaiyan Chang, Zhuolin Li, Xinyang He, Chujie Chen, Cangyuan Li, Mengdi Wang, Haobo Xu, Yinhe Han, Ying Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04736)  

**Abstract**: Large Language Models (LLMs) show significant potential for automating Register-Transfer Level (RTL) code generation. However, current approaches face a critical challenge: they can not simultaneously optimize for functional correctness and hardware quality (Power, Performance, Area - PPA). Methods based on supervised fine-tuning often generate functionally correct but PPA-suboptimal code, lacking mechanisms to learn optimization principles. In contrast, post-processing techniques that attempt to improve PPA metrics after generation are often inefficient because they operate externally without updating the LLM's parameters, thus failing to enhance the model's intrinsic design capabilities.
To bridge this gap, we introduce ChipSeek-R1, a hierarchical reward-driven reinforcement learning framework to train LLMs to generate RTL code that achieves both functional correctness and optimized PPA metrics. ChipSeek-R1 employs a hierarchical reward system, which incorporates direct feedback on syntax, functional correctness (from simulators) and PPA metrics (from synthesis tools) during reinforcement learning. This enables the model to learn complex hardware design trade-offs via trial-and-error, generating RTL code that is both functionally correct and PPA-optimized. Evaluating ChipSeek-R1 on standard benchmarks (VerilogEval, RTLLM), we achieve state-of-the-art results in functional correctness. Notably, on the RTLLM benchmark, ChipSeek-R1 generated 27 RTL designs surpassing the PPA metrics of the original human-written code. Our findings demonstrate the effectiveness of integrating toolchain feedback into LLM training and highlight the potential for reinforcement learning to enable automated generation of human-surpassing RTL code. We open-source our code in anonymous github. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自动化寄存传输级（RTL）代码生成方面显示出显著潜力。然而，当前方法面临一个关键挑战：它们无法同时优化功能正确性和硬件质量（功率、性能、面积 - PPA）。基于监督微调的方法往往生成功能正确但PPA次优的代码，缺乏学习优化原则的机制。相比之下，试图在生成后改进PPA指标的后处理技术通常效率低下，因为它们在外部分析而不更新LLM的参数，因此无法增强模型的内在设计能力。为了避免这一差距，我们提出了一种分层奖励驱动的强化学习框架ChipSeek-R1，以训练LLM生成同时实现功能正确性和优化PPA指标的RTL代码。ChipSeek-R1采用分层奖励系统，在强化学习过程中 Incorporates 对语法、功能正确性（来自模拟器）和PPA指标（来自综合工具）的直接反馈，使模型能够通过试错学习复杂的硬件设计权衡，生成既功能正确又PPA优化的RTL代码。在标准基准测试（VerilogEval, RTLLM）上评估ChipSeek-R1，我们在功能正确性方面取得了最先进的结果。值得注意的是，在RTLLM基准测试中，ChipSeek-R1生成了27种超过原始人工编写代码PPA指标的RTL设计。我们的研究结果展示了将工具链反馈集成到LLM训练中的有效性，并突显了强化学习在自动化生成超越人类的RTL代码方面具有潜力。我们已将代码开源在匿名GitHub上。 

---
# LumiCRS: Asymmetric Contrastive Prototype Learning for Long-Tail Conversational Movie Recommendation 

**Title (ZH)**: LumiCRS: 不对称对比原型学习在长尾对话电影推荐中的应用 

**Authors**: Jinzhi Wang, Bin Li, Qingke Peng, Haozhou Li, Zeyuan Zeng, Ruimeng Li, Biyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.04722)  

**Abstract**: Conversational recommender systems (CRSs) often suffer from an extreme long-tail distribution of dialogue data, causing a strong bias toward head-frequency blockbusters that sacrifices diversity and exacerbates the cold-start problem. An empirical analysis of DCRS and statistics on the REDIAL corpus show that only 10% of head movies account for nearly half of all mentions, whereas about 70% of tail movies receive merely 26% of the attention. This imbalance gives rise to three critical challenges: head over-fitting, body representation drift, and tail sparsity. To address these issues, we propose LumiCRS, an end-to-end framework that mitigates long-tail imbalance through three mutually reinforcing layers: (i) an Adaptive Comprehensive Focal Loss (ACFL) that dynamically adjusts class weights and focusing factors to curb head over-fitting and reduce popularity bias; (ii) Prototype Learning for Long-Tail Recommendation, which selects semantic, affective, and contextual prototypes to guide clustering and stabilize body and tail representations; and (iii) a GPT-4o-driven prototype-guided dialogue augmentation module that automatically generates diverse long-tail conversational snippets to alleviate tail sparsity and distribution shift. Together, these strategies enable LumiCRS to markedly improve recommendation accuracy, diversity, and fairness: on the REDIAL and INSPIRED benchmarks, LumiCRS boosts Recall@10 and Tail-Recall@10 by 7-15% over fifteen strong baselines, while human evaluations confirm superior fluency, informativeness, and long-tail relevance. These results demonstrate the effectiveness of multi-layer collaboration in building an efficient and fair long-tail conversational recommender. 

**Abstract (ZH)**: 长尾对话数据分布不平衡的会话推荐系统（CRSs）往往受到头部效应的严重影响，导致高度偏向于高频率的头部项目，牺牲了多样性并加剧了冷启动问题。对DCRS的实证分析和REDIAL语料库的统计数据表明，只有10%的头部电影占据了近一半的提及次数，而大约70%的尾部电影仅获得了26%的关注度。这种不平衡引发了三个关键挑战：头部过拟合、身体表示漂移和尾部稀疏性。为了解决这些问题，我们提出了LumiCRS，这是一种端到端框架，通过三层相互加强的机制缓解长尾不平衡：（i）自适应全面焦点损失（ACFL），动态调整类别权重和焦点因子以抑制头部过拟合并减少流行度偏见；（ii）长尾推荐的原型学习，选择语义、情感和上下文原型来指导聚类并稳定身体和尾部表示；以及（iii）由GPT-4o驱动的基于原型的对话增强模块，它可以自动生成多样化的长尾对话片段，以减轻尾部稀疏性和分布偏移。这些策略使得LumiCRS显著提高了推荐精度、多样性和公平性：在REDIAL和INSPIRED基准上，LumiCRS相比十五个强劲的基线，在Recall@10和Tail-Recall@10上提高了7-15%，而人工评估证实了其在流畅性、信息量和长尾相关性方面的优越性。这些结果证明了在构建高效的公平长尾会话推荐系统中多层协作的有效性。 

---
# Advocate for Complete Benchmarks for Formal Reasoning with Formal/Informal Statements and Formal/Informal Proofs 

**Title (ZH)**: 倡导完整的基准以形形兼备的陈述和证明进行形式化推理 

**Authors**: Roozbeh Yousefzadeh, Xuenan Cao  

**Link**: [PDF](https://arxiv.org/pdf/2507.04719)  

**Abstract**: This position paper provides a critical but constructive discussion of current practices in benchmarking and evaluative practices in the field of formal reasoning and automated theorem proving. We take the position that open code, open data, and benchmarks that are complete and error-free will accelerate progress in this field. We identify practices that create barriers to contributing to this field and suggest ways to remove them. We also discuss some of the practices that might produce misleading evaluative information. We aim to create discussions that bring together people from various groups contributing to automated theorem proving, autoformalization, and informal reasoning. 

**Abstract (ZH)**: 这篇立场论文对形式推理和自动定理证明领域当前的基准测试和评估实践进行了批判性但建设性的讨论。我们认为，开源代码、开源数据以及完整无误的基准测试将加速该领域的发展。我们指出了阻碍贡献该领域的实践，并提出了去除这些障碍的方法。同时，我们讨论了一些可能导致误导性评价信息的做法。我们的目标是促进来自自动化定理证明、自动形式化和非形式推理等领域贡献者的讨论。 

---
# Trojan Horse Prompting: Jailbreaking Conversational Multimodal Models by Forging Assistant Message 

**Title (ZH)**: 木马匹诺特攻击：通过伪造助手消息解锁对话型多模态模型 

**Authors**: Wei Duan, Li Qian  

**Link**: [PDF](https://arxiv.org/pdf/2507.04673)  

**Abstract**: The rise of conversational interfaces has greatly enhanced LLM usability by leveraging dialogue history for sophisticated reasoning. However, this reliance introduces an unexplored attack surface. This paper introduces Trojan Horse Prompting, a novel jailbreak technique. Adversaries bypass safety mechanisms by forging the model's own past utterances within the conversational history provided to its API. A malicious payload is injected into a model-attributed message, followed by a benign user prompt to trigger harmful content generation. This vulnerability stems from Asymmetric Safety Alignment: models are extensively trained to refuse harmful user requests but lack comparable skepticism towards their own purported conversational history. This implicit trust in its "past" creates a high-impact vulnerability. Experimental validation on Google's Gemini-2.0-flash-preview-image-generation shows Trojan Horse Prompting achieves a significantly higher Attack Success Rate (ASR) than established user-turn jailbreaking methods. These findings reveal a fundamental flaw in modern conversational AI security, necessitating a paradigm shift from input-level filtering to robust, protocol-level validation of conversational context integrity. 

**Abstract (ZH)**: 对话界面的兴起通过利用对话历史增强了大语言模型的可用性，进行复杂的推理。然而，这种依赖引入了一个未被探索的攻击面。本文介绍了特洛伊木马提示技术，这是一种新颖的 Jailbreak 技术。攻击者通过在提供给模型 API 的对话历史中伪造模型自身的过往陈述，绕过安全机制。恶意负载被注入到一个归因于模型的消息中，随后是一个看似无害的用户提示，以触发有害内容的生成。这种漏洞源自不对称的安全对齐：模型被广泛训练以拒绝有害的用户请求，但缺乏对其自身声称的对话历史的类似怀疑。这种对“过去”的隐含信任创造了高影响的漏洞。在 Google 的 Gemini-2.0-flash-preview-image-generation 上的实验验证表明，特洛伊木马提示技术的攻击成功率 (ASR) 显著高于现有的用户回合 Jailbreak 方法。这些发现揭示了现代对话 AI 安全中的根本缺陷，需要从输入级过滤转向对对话上下文完整性的 robust 协议级验证。 

---
# Can Prompt Difficulty be Online Predicted for Accelerating RL Finetuning of Reasoning Models? 

**Title (ZH)**: 基于提示难度的在线预测以加速逻辑推理模型的RL微调是否可行？ 

**Authors**: Yun Qu, Qi Cheems Wang, Yixiu Mao, Vincent Tao Hu, Xiangyang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2507.04632)  

**Abstract**: Recent advances have witnessed the effectiveness of reinforcement learning (RL) finetuning in enhancing the reasoning capabilities of large language models (LLMs). The optimization process often requires numerous iterations to achieve satisfactory performance, resulting in high computational costs due to the need for frequent prompt evaluations under intensive LLM interactions and repeated policy updates. Appropriate online prompt selection methods reduce iteration steps by prioritizing informative prompts during training, while the pipeline's reliance on exhaustive prompt evaluation and subset selection for optimization still incurs substantial computational overhead due to frequent LLM inference calls. Distinguished from these direct evaluate-then-select schemes, this work investigates iterative approximate evaluation for arbitrary prompts and introduces Model Predictive Prompt Selection (MoPPS), a Bayesian risk-predictive framework that online estimates prompt difficulty without requiring costly LLM interactions. Technically, MoPPS models each prompt's success rate as a latent variable, performs streaming Bayesian inference, and employs posterior sampling in a constructed multi-armed bandit machine, enabling sample efficient and adaptive prompt selection. Extensive experiments across mathematics, planning, and vision-based geometry tasks show that MoPPS reliably predicts prompt difficulty and accelerates training with significantly reduced LLM rollouts. 

**Abstract (ZH)**: 近期研究见证了强化学习（RL）微调在增强大型语言模型（LLMs）推理能力方面的有效性。优化过程通常需要多次迭代以达到满意的性能，由于需要在密集的LLM交互和反复的策略更新中频繁进行提示评估，因此产生了高计算成本。适当的在线提示选择方法通过优先选择有信息性的提示来减少迭代步骤，但管道依赖于详尽的提示评估和子集选择的优化仍然会因频繁的LLM推理调用而产生巨大的计算开销。不同于这些直接评估-然后选择的方案，本工作研究了任意提示的迭代近似评估，并引入了模型预测提示选择（MoPPS），这是一种贝叶斯风险预测框架，可以在不进行昂贵的LLM交互的情况下在线估计提示难度。技术上，MoPPS 将每个提示的成功率建模为潜在变量，执行流式贝叶斯推理，并在构建的多臂槽机中使用后验采样，从而实现样本高效和自适应的提示选择。广泛的实验涵盖了数学、规划和基于视觉的几何任务，表明MoPPS 可靠地预测了提示难度，并显著减少了LLM滚动次数，加速了训练。 

---
# DisMS-TS: Eliminating Redundant Multi-Scale Features for Time Series Classification 

**Title (ZH)**: DisMS-TS: 消除时间序列分类中的冗余多尺度特征 

**Authors**: Zhipeng Liu, Peibo Duan, Binwu Wang, Xuan Tang, Qi Chu, Changsheng Zhang, Yongsheng Huang, Bin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04600)  

**Abstract**: Real-world time series typically exhibit complex temporal variations, making the time series classification task notably challenging. Recent advancements have demonstrated the potential of multi-scale analysis approaches, which provide an effective solution for capturing these complex temporal patterns. However, existing multi-scale analysis-based time series prediction methods fail to eliminate redundant scale-shared features across multi-scale time series, resulting in the model over- or under-focusing on scale-shared features. To address this issue, we propose a novel end-to-end Disentangled Multi-Scale framework for Time Series classification (DisMS-TS). The core idea of DisMS-TS is to eliminate redundant shared features in multi-scale time series, thereby improving prediction performance. Specifically, we propose a temporal disentanglement module to capture scale-shared and scale-specific temporal representations, respectively. Subsequently, to effectively learn both scale-shared and scale-specific temporal representations, we introduce two regularization terms that ensure the consistency of scale-shared representations and the disparity of scale-specific representations across all temporal scales. Extensive experiments conducted on multiple datasets validate the superiority of DisMS-TS over its competitive baselines, with the accuracy improvement up to 9.71%. 

**Abstract (ZH)**: Real-world时间序列通常表现出复杂的时变特性，使得时间序列分类任务尤为具有挑战性。近年来的研究表明，多尺度分析方法具有潜在的价值，能够有效捕获这些复杂的时变模式。然而，现有的基于多尺度分析的时间序列预测方法未能消除多尺度时间序列中的冗余共享特征，导致模型过度或不足地关注共享特征。为了解决这一问题，我们提出了一种新的端到端解耦多尺度框架（DisMS-TS）用于时间序列分类。DisMS-TS的核心思想是消除多尺度时间序列中的冗余共享特征，从而提高预测性能。具体而言，我们提出了一种时序解耦模块分别捕获多尺度共享和多尺度特定的时间表示。随后，为了有效地学习多尺度共享和特定的时间表示，我们引入了两种正则化项，以确保所有时序尺度上共享表示的一致性和特定表示的差异性。在多个数据集上进行的广泛实验验证了DisMS-TS在多个基准方法中的优越性，准确率提高了高达9.71%。 

---
# Exploring Core and Periphery Precepts in Biological and Artificial Intelligence: An Outcome-Based Perspective 

**Title (ZH)**: 基于结果导向视角探索生物学与人工智能的核心与边缘原则 

**Authors**: Niloofar Shadab, Tyler Cody, Alejandro Salado, Taylan G. Topcu, Mohammad Shadab, Peter Beling  

**Link**: [PDF](https://arxiv.org/pdf/2507.04594)  

**Abstract**: Engineering methodologies predominantly revolve around established principles of decomposition and recomposition. These principles involve partitioning inputs and outputs at the component level, ensuring that the properties of individual components are preserved upon composition. However, this view does not transfer well to intelligent systems, particularly when addressing the scaling of intelligence as a system property. Our prior research contends that the engineering of general intelligence necessitates a fresh set of overarching systems principles. As a result, we introduced the "core and periphery" principles, a novel conceptual framework rooted in abstract systems theory and the Law of Requisite Variety. In this paper, we assert that these abstract concepts hold practical significance. Through empirical evidence, we illustrate their applicability to both biological and artificial intelligence systems, bridging abstract theory with real-world implementations. Then, we expand on our previous theoretical framework by mathematically defining core-dominant vs periphery-dominant systems. 

**Abstract (ZH)**: 工程方法主要围绕分解与重组的既定原则进行。这些原则涉及在组件级别划分输入和输出，并确保在组合时个体组件的属性得以保留。然而，这种观点在处理智能系统中的智能扩展这一系统属性时并不适用。我们此前的研究认为，通用人工智能的工程需要一套新的综合系统原则。因此，我们引入了“核心和外围”原则，这是一个基于抽象系统理论和必要的变异性定律的新颖概念框架。在本文中，我们认为这些抽象概念具有实际意义。通过实验证据，我们展示了它们在生物学和人工智能系统中的适用性，将抽象理论与实际应用相结合。然后，我们通过数学定义进一步扩展了我们之前理论框架，区分核心主导系统与外围主导系统。 

---
# Towards integration of Privacy Enhancing Technologies in Explainable Artificial Intelligence 

**Title (ZH)**: 向Privacy Enhancing Technologies与Explainable Artificial Intelligence的集成方向探索 

**Authors**: Sonal Allana, Rozita Dara, Xiaodong Lin, Pulei Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2507.04528)  

**Abstract**: Explainable Artificial Intelligence (XAI) is a crucial pathway in mitigating the risk of non-transparency in the decision-making process of black-box Artificial Intelligence (AI) systems. However, despite the benefits, XAI methods are found to leak the privacy of individuals whose data is used in training or querying the models. Researchers have demonstrated privacy attacks that exploit explanations to infer sensitive personal information of individuals. Currently there is a lack of defenses against known privacy attacks targeting explanations when vulnerable XAI are used in production and machine learning as a service system. To address this gap, in this article, we explore Privacy Enhancing Technologies (PETs) as a defense mechanism against attribute inference on explanations provided by feature-based XAI methods. We empirically evaluate 3 types of PETs, namely synthetic training data, differentially private training and noise addition, on two categories of feature-based XAI. Our evaluation determines different responses from the mitigation methods and side-effects of PETs on other system properties such as utility and performance. In the best case, PETs integration in explanations reduced the risk of the attack by 49.47%, while maintaining model utility and explanation quality. Through our evaluation, we identify strategies for using PETs in XAI for maximizing benefits and minimizing the success of this privacy attack on sensitive personal information. 

**Abstract (ZH)**: 可解释的人工智能（XAI）是减轻黑盒人工智能（AI）系统决策过程不透明性风险的关键途径。然而，尽管XAI方法带来了好处，这些方法也被发现泄露了用于训练或查询模型的个体隐私信息。研究人员已证明利用解释来进行隐私攻击，以推断个体的敏感个人信息。目前，尚未针对使用可能存在漏洞的XAI方法进行隐私攻击的防御措施，特别是在生产环境和机器学习即服务系统中。为解决这一问题，本文探讨了隐私增强技术（PETs）作为特征基于XAI方法提供的解释进行属性推断的一种防御机制。我们通过实证评估了三种类型的PETs，即合成训练数据、差分隐私训练和噪声添加，对两类特征基于XAI方法进行评估。评估结果确定了不同缓解方法和PETs对其他系统属性（如效用和性能）的副作用。在最佳情况下，PETs整合到解释中可将攻击风险降低49.47%，同时保持模型效用和解释质量。通过我们的评估，我们确定了在XAI中使用PETs来最大化利益并最小化此类隐私攻击成功的策略。 

---
# Churn-Aware Recommendation Planning under Aggregated Preference Feedback 

**Title (ZH)**: 基于聚合偏好反馈的 churn 意识推荐规划 

**Authors**: Gur Keinan, Omer Ben-Porat  

**Link**: [PDF](https://arxiv.org/pdf/2507.04513)  

**Abstract**: We study a sequential decision-making problem motivated by recent regulatory and technological shifts that limit access to individual user data in recommender systems (RSs), leaving only population-level preference information. This privacy-aware setting poses fundamental challenges in planning under uncertainty: Effective personalization requires exploration to infer user preferences, yet unsatisfactory recommendations risk immediate user churn. To address this, we introduce the Rec-APC model, in which an anonymous user is drawn from a known prior over latent user types (e.g., personas or clusters), and the decision-maker sequentially selects items to recommend. Feedback is binary -- positive responses refine the posterior via Bayesian updates, while negative responses result in the termination of the session.
We prove that optimal policies converge to pure exploitation in finite time and propose a branch-and-bound algorithm to efficiently compute them. Experiments on synthetic and MovieLens data confirm rapid convergence and demonstrate that our method outperforms the POMDP solver SARSOP, particularly when the number of user types is large or comparable to the number of content categories. Our results highlight the applicability of this approach and inspire new ways to improve decision-making under the constraints imposed by aggregated preference data. 

**Abstract (ZH)**: 我们研究一个由近期监管和技术变化引起的序贯决策问题，这些变化限制了推荐系统（RSs）中个体用户数据的访问，仅留下群体级别的偏好信息。这种隐私意识设定在不确定性的计划中提出了根本性的挑战：有效的个性化需要探索以推断用户偏好，但不满意的推荐可能会导致用户的即时流失。为了解决这一问题，我们引入了Rec-APC模型，在该模型中，匿名用户来自已知的潜在用户类型先验分布（例如，角色或聚类），决策者随后顺序选择推荐项目。反馈为二元的——正面响应通过贝叶斯更新精炼后验概率，而负面响应导致会话终止。我们证明了最优策略在有限时间内收敛到纯粹的利用，并提出了一种分支定界算法来高效地计算这些策略。实验结果表明，我们的方法在合成数据和MovieLens数据上表现出快速收敛，并且在潜在用户类型数量大或与内容类别数量相当时，优于POMDP求解器SARSOP。我们的结果强调了该方法的应用性和在受限于聚合偏好数据的情况下改进决策的新途径。 

---
# Thousand-Brains Systems: Sensorimotor Intelligence for Rapid, Robust Learning and Inference 

**Title (ZH)**: 千脑系统：感知运动智能以实现快速稳健的学习与推理 

**Authors**: Niels Leadholm, Viviane Clay, Scott Knudstrup, Hojae Lee, Jeff Hawkins  

**Link**: [PDF](https://arxiv.org/pdf/2507.04494)  

**Abstract**: Current AI systems achieve impressive performance on many tasks, yet they lack core attributes of biological intelligence, including rapid, continual learning, representations grounded in sensorimotor interactions, and structured knowledge that enables efficient generalization. Neuroscience theory suggests that mammals evolved flexible intelligence through the replication of a semi-independent, sensorimotor module, a functional unit known as a cortical column. To address the disparity between biological and artificial intelligence, thousand-brains systems were proposed as a means of mirroring the architecture of cortical columns and their interactions.
In the current work, we evaluate the unique properties of Monty, the first implementation of a thousand-brains system. We focus on 3D object perception, and in particular, the combined task of object recognition and pose estimation. Utilizing the YCB dataset of household objects, we first assess Monty's use of sensorimotor learning to build structured representations, finding that these enable robust generalization. These representations include an emphasis on classifying objects by their global shape, as well as a natural ability to detect object symmetries. We then explore Monty's use of model-free and model-based policies to enable rapid inference by supporting principled movements. We find that such policies complement Monty's modular architecture, a design that can accommodate communication between modules to further accelerate inference speed via a novel `voting' algorithm. Finally, we examine Monty's use of associative, Hebbian-like binding to enable rapid, continual, and computationally efficient learning, properties that compare favorably to current deep learning architectures. While Monty is still in a nascent stage of development, these findings support thousand-brains systems as a powerful and promising new approach to AI. 

**Abstract (ZH)**: 当前的AI系统在许多任务上取得了令人印象深刻的性能，但缺乏生物智能的核心属性，包括快速、持续的学习，基于传感器动作交互的表示，以及能够高效泛化的结构化知识。神经科学理论表明，哺乳动物通过复制半独立的、传感器动作模块，即脑皮层柱这一功能单位，进化出了灵活的智能。为了缩小生物智能与人工智能之间的差距， thousand-brains 系统被提出作为模仿脑皮层柱及其交互架构的一种手段。
在当前的研究中，我们评估了Monty的特点，Monty是第一个实现thousand-brains系统的实例。我们集中在三维物体感知，尤其是物体识别和姿态估计的综合任务上。利用家庭用品的YCB数据集，我们首先评估Monty如何利用传感器动作学习构建结构化的表示，发现这些表示能够实现稳健的泛化。这些表示包括强调通过整体形状分类物体，并且自然具备检测物体对称性的能力。然后，我们探讨了Monty如何利用无模型和基于模型的策略来实现快速推理，通过支持原理性的运动来支持。我们发现，这些策略与Monty的模块化架构相补充，这种设计可以通过一种新颖的“投票”算法在模块之间进行通信，进一步加速推理速度。最后，我们研究了Monty利用类似艾宾浩斯绑定来实现快速、持续、计算高效的学习，这些特性与当前的深度学习架构相比具有优势。尽管Monty仍处于发展的初级阶段，但这些发现支持thousand-brains系统作为一种强大而有前途的新AI方法。 

---
# Anomalous Decision Discovery using Inverse Reinforcement Learning 

**Title (ZH)**: 使用逆强化学习发现异常决策 

**Authors**: Ashish Bastola, Mert D. Pesé, Long Cheng, Jonathon Smereka, Abolfazl Razi  

**Link**: [PDF](https://arxiv.org/pdf/2507.04464)  

**Abstract**: Anomaly detection plays a critical role in Autonomous Vehicles (AVs) by identifying unusual behaviors through perception systems that could compromise safety and lead to hazardous situations. Current approaches, which often rely on predefined thresholds or supervised learning paradigms, exhibit reduced efficacy when confronted with unseen scenarios, sensor noise, and occlusions, leading to potential safety-critical failures. Moreover, supervised methods require large annotated datasets, limiting their real-world feasibility. To address these gaps, we propose an anomaly detection framework based on Inverse Reinforcement Learning (IRL) to infer latent driving intentions from sequential perception data, thus enabling robust identification. Specifically, we present Trajectory-Reward Guided Adaptive Pre-training (TRAP), a novel IRL framework for anomaly detection, to address two critical limitations of existing methods: noise robustness and generalization to unseen scenarios. Our core innovation is implicitly learning temporal credit assignments via reward and worst-case supervision. We leverage pre-training with variable-horizon sampling to maximize time-to-consequence, resulting in early detection of behavior deviation. Experiments on 14,000+ simulated trajectories demonstrate state-of-the-art performance, achieving 0.90 AUC and 82.2\% F1-score - outperforming similarly trained supervised and unsupervised baselines by 39\% on Recall and 12\% on F1-score, respectively. Similar performance is achieved while exhibiting robustness to various noise types and generalization to unseen anomaly types. Our code will be available at: this https URL 

**Abstract (ZH)**: 自主驾驶中的异常检测通过感知系统识别可能危及安全的异常行为，发挥着关键作用。当前的方法通常依赖预定义的阈值或监督学习范式，在遇到未知场景、传感器噪声和遮挡时效用降低，可能导致潜在的安全关键性故障。此外，监督方法需要大规模标注数据集，限制了其实用性。为解决这些问题，我们提出了一种基于逆强化学习（IRL）的异常检测框架，以从序列感知数据中推断隐含的驾驶意图，从而实现稳健的识别。具体来说，我们呈现了轨迹-奖励引导自适应预训练（TRAP），这是一种新颖的用于异常检测的IRL框架，解决了现有方法的两个关键问题：噪声鲁棒性和对未知场景的泛化能力。我们的核心创新是通过奖励和最坏情况监督隐式学习时间信用分配。我们利用变时窗采样的预训练以最大化后果时间，实现行为偏离的早期检测。在14,000多个模拟轨迹上的实验展示了最先进的性能，AUC达到0.90，F1分数为82.2%，分别比同样训练的监督和无监督基线在召回率上高出39%、F1分数上高出12%。同时，该方法在面对不同类型的噪声和未见过的异常类型时表现出鲁棒性。代码将在此处提供：this https URL 

---
# A Linguistic Analysis of Spontaneous Thoughts: Investigating Experiences of Déjà Vu, Unexpected Thoughts, and Involuntary Autobiographical Memories 

**Title (ZH)**: 自发思维的语言分析：探讨 déjà vu、意外思绪及不自主自传记忆的经验 

**Authors**: Videep Venkatesha, Mary Cati Poulos, Christopher Steadman, Caitlin Mills, Anne M. Cleary, Nathaniel Blanchard  

**Link**: [PDF](https://arxiv.org/pdf/2507.04439)  

**Abstract**: The onset of spontaneous thoughts are reflective of dynamic interactions between cognition, emotion, and attention. Typically, these experiences are studied through subjective appraisals that focus on their triggers, phenomenology, and emotional salience. In this work, we use linguistic signatures to investigate Deja Vu, Involuntary Autobiographical Memories and Unexpected Thoughts. Specifically, we analyze the inherent characteristics of the linguistic patterns in participant generated descriptions of these thought types. We show how, by positioning language as a window into spontaneous cognition, existing theories on these attentional states can be updated and reaffirmed. Our findings align with prior research, reinforcing that Deja Vu is a metacognitive experience characterized by abstract and spatial language, Involuntary Autobiographical Memories are rich in personal and emotionally significant detail, and Unexpected Thoughts are marked by unpredictability and cognitive disruption. This work is demonstrative of languages potential to reveal deeper insights into how internal spontaneous cognitive states manifest through expression. 

**Abstract (ZH)**: 自发思维的 onset 反映了认知、情感和注意力之间的动态交互。通常，这些体验通过关注触发因素、现象学和情绪显著性来主观评估。在本文中，我们使用语言特征来探究 Déjà Vu、不自愿的自传体记忆和意外思维。具体而言，我们分析了参与者对这些思维类型生成描述的固有语言模式特征。我们通过将语言定位为了解自发认知的窗户，展示了如何更新和完善现有注意力状态的理论。我们的发现与先前的研究一致，证实了 Déjà Vu 是一种元认知体验，特征为抽象和空间语言，不自愿的自传体记忆富含个人和情感意义的细节，意外思维则表现出不可预测性和认知中断的特点。这项工作展示了语言揭示内部自发认知状态表达方式中更深层次见解的潜力。 

---
# MedGellan: LLM-Generated Medical Guidance to Support Physicians 

**Title (ZH)**: MedGellan: 由大语言模型生成的医疗指导以支持医师 

**Authors**: Debodeep Banerjee, Burcu Sayin, Stefano Teso, Andrea Passerini  

**Link**: [PDF](https://arxiv.org/pdf/2507.04431)  

**Abstract**: Medical decision-making is a critical task, where errors can result in serious, potentially life-threatening consequences. While full automation remains challenging, hybrid frameworks that combine machine intelligence with human oversight offer a practical alternative. In this paper, we present MedGellan, a lightweight, annotation-free framework that uses a Large Language Model (LLM) to generate clinical guidance from raw medical records, which is then used by a physician to predict diagnoses. MedGellan uses a Bayesian-inspired prompting strategy that respects the temporal order of clinical data. Preliminary experiments show that the guidance generated by the LLM with MedGellan improves diagnostic performance, particularly in recall and $F_1$ score. 

**Abstract (ZH)**: 医学决策是一项关键任务，其中的错误可能导致严重的、甚至危及生命的结果。虽然完全自动化仍具有挑战性，但将机器智能与人类监督相结合的混合框架提供了实用的替代方案。在本文中，我们介绍了MedGellan，这是一种轻量级、无标注的框架，利用大型语言模型（LLM）从原始医疗记录中生成临床指导，供医生用于预测诊断。MedGellan 使用一种受贝叶斯启发的提示策略，尊重临床数据的时间顺序。初步实验表明，MedGellan 生成的指导提高了诊断性能，尤其是在召回率和 F1 分数方面。 

---
# ARMR: Adaptively Responsive Network for Medication Recommendation 

**Title (ZH)**: ARMR：自适应响应网络在药物推荐中的应用 

**Authors**: Feiyue Wu, Tianxing Wu, Shenqi Jing  

**Link**: [PDF](https://arxiv.org/pdf/2507.04428)  

**Abstract**: Medication recommendation is a crucial task in healthcare, especially for patients with complex medical conditions. However, existing methods often struggle to effectively balance the reuse of historical medications with the introduction of new drugs in response to the changing patient conditions. In order to address this challenge, we propose an Adaptively Responsive network for Medication Recommendation (ARMR), a new method which incorporates 1) a piecewise temporal learning component that distinguishes between recent and distant patient history, enabling more nuanced temporal understanding, and 2) an adaptively responsive mechanism that dynamically adjusts attention to new and existing drugs based on the patient's current health state and medication history. Experiments on the MIMIC-III and MIMIC-IV datasets indicate that ARMR has better performance compared with the state-of-the-art baselines in different evaluation metrics, which contributes to more personalized and accurate medication recommendations. The source code is publicly avaiable at: this https URL. 

**Abstract (ZH)**: 适应性响应网络在医疗药物推荐中的应用：一种新的方法 

---
# LayerCake: Token-Aware Contrastive Decoding within Large Language Model Layers 

**Title (ZH)**: LayerCake: 层内大型语言模型中具有标记意识的对比解码 

**Authors**: Jingze Zhu, Yongliang Wu, Wenbo Zhu, Jiawang Cao, Yanqiang Zheng, Jiawei Chen, Xu Yang, Bernt Schiele, Jonas Fischer, Xinting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04404)  

**Abstract**: Large language models (LLMs) excel at natural language understanding and generation but remain vulnerable to factual errors, limiting their reliability in knowledge-intensive tasks. While decoding-time strategies provide a promising efficient solution without training, existing methods typically treat token-level and layer-level signals in isolation, overlooking the joint dynamics between them. In this work, we introduce a token-aware, layer-localized contrastive decoding method that aligns specific token types with their most influential transformer layers to improve factual generation. Through empirical attention analysis, we identify two key patterns: punctuation tokens receive dominant attention in early layers, while conceptual tokens govern semantic reasoning in intermediate layers. By selectively suppressing attention to these token types at their respective depths, we achieve the induction of controlled factual degradation and derive contrastive signals to guide the final factual decoding. Our method requires no additional training or model modification, and experiments demonstrate that our method consistently improves factuality across multiple LLMs and various benchmarks. 

**Abstract (ZH)**: 大语言模型（LLMs）在自然语言理解与生成方面表现出色，但在事实准确性上仍存在局限，限制了其在知识密集型任务中的可靠性。尽管解码时的策略可以提供一种无需额外训练的有效解决方案，但现有方法通常将标记级和层级信号隔离开来，忽略了它们之间的联合动态。在本工作中，我们引入了一种标记感知、层局部的对比解码方法，将特定标记类型与其最具影响力的变换器层对齐，以提升事实生成的准确性。通过实证注意力分析，我们发现两种关键模式：标点符号标记在早期层中占据主导注意力，而概念性标记在中间层中控制语义推理。通过在相应深度选择性抑制这些标记类型的注意力，我们实现了可控事实退化的诱导，并提取对比信号引导最终的事实解码。该方法不需要额外的训练或模型修改，实验证明我们的方法在多个大语言模型和各种基准测试中一致提升了事实准确性。 

---
# DC-Mamber: A Dual Channel Prediction Model based on Mamba and Linear Transformer for Multivariate Time Series Forecasting 

**Title (ZH)**: DC-Mamber：基于Mamba和线性变压器的双通道预测模型多变量时间序列预测 

**Authors**: Bing Fan, Shusen Ma, Yun-Bo Zhao, Yu Kang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04381)  

**Abstract**: In multivariate time series forecasting (MTSF), existing strategies for processing sequences are typically categorized as channel-independent and channel-mixing. The former treats all temporal information of each variable as a token, focusing on capturing local temporal features of individual variables, while the latter constructs a token from the multivariate information at each time step, emphasizing the modeling of global temporal dependencies. Current mainstream models are mostly based on Transformer and the emerging Mamba. Transformers excel at modeling global dependencies through self-attention mechanisms but exhibit limited sensitivity to local temporal patterns and suffer from quadratic computational complexity, restricting their efficiency in long-sequence processing. In contrast, Mamba, based on state space models (SSMs), achieves linear complexity and efficient long-range modeling but struggles to aggregate global contextual information in parallel. To overcome the limitations of both models, we propose DC-Mamber, a dual-channel forecasting model based on Mamba and linear Transformer for time series forecasting. Specifically, the Mamba-based channel employs a channel-independent strategy to extract intra-variable features, while the Transformer-based channel adopts a channel-mixing strategy to model cross-timestep global dependencies. DC-Mamber first maps the raw input into two distinct feature representations via separate embedding layers. These representations are then processed by a variable encoder (built on Mamba) and a temporal encoder (built on linear Transformer), respectively. Finally, a fusion layer integrates the dual-channel features for prediction. Extensive experiments on eight public datasets confirm DC-Mamber's superior accuracy over existing models. 

**Abstract (ZH)**: 基于Mamba和线性Transformer的双通道时间序列预测模型DC-Mamber 

---
# MOD-X: A Modular Open Decentralized eXchange Framework proposal for Heterogeneous Interoperable Artificial Agents 

**Title (ZH)**: MOD-X: 一种模块化开放去中心化异构可互操作人工代理交易平台框架提案 

**Authors**: Georgios Ioannides, Christos Constantinou, Vinija Jain, Aman Chadha, Aaron Elkins  

**Link**: [PDF](https://arxiv.org/pdf/2507.04376)  

**Abstract**: As Artificial Intelligence systems evolve from monolithic models to ecosystems of specialized agents, the need for standardized communication protocols becomes increasingly critical. This paper introduces MOD-X (Modular Open Decentralized eXchange), a novel architectural framework proposal for agent interoperability that addresses key limitations of existing protocols. Unlike current approaches, MOD-X proposes a layered architecture with a Universal Message Bus, thorough state management, translation capabilities, and blockchain-based security mechanisms. We present MOD-X's architecture, compare it with existing protocols, and demonstrate its application through a worked example how it enables integration between heterogeneous specialist agents (agents with different architectures, vendors, capabilities, and knowledge representations--including rule-based systems, neural networks, symbolic reasoning engines, and legacy software with agent wrappers). MOD-X's key innovations include a publish-subscribe communication model, semantic capability discovery, and dynamic workflow orchestration--providing a framework that bridges theoretical formalism with practical implementation. This architecture addresses the growing need for truly decentralized, interoperable agent ecosystems that can scale effectively without the need for central coordination. 

**Abstract (ZH)**: 随着人工智能系统从单一模型进化为专门代理的生态系统，标准化通信协议的需求变得越来越关键。本文介绍了MOD-X（模块化开放去中心化交换），这是一种针对现有协议关键局限性的新型架构框架提议，旨在实现代理互操作性。MOD-X提出了一种分层架构，包括通用消息总线、全面的状态管理、翻译能力以及基于区块链的安全机制。我们介绍了MOD-X的架构，将其与现有协议进行比较，并通过一个工作示例展示了它如何使不同架构、供应商、能力和知识表示的异质专业代理（包括基于规则的系统、神经网络、符号推理引擎以及带有代理封装的遗留软件）之间的集成成为可能。MOD-X的关键创新包括发布/订阅通信模型、语义能力发现以及动态工作流编排，提供了一种将理论形式化与实际实施相结合的框架。该架构解决了真正去中心化、互操作性强的代理生态系统需要有效扩展而无需中央协调的日益增长的需求。 

---
# WebSynthesis: World-Model-Guided MCTS for Efficient WebUI-Trajectory Synthesis 

**Title (ZH)**: WebSynthesis: 世界模型引导的 Monte Carlo 森林搜索高效网页UI轨迹合成 

**Authors**: Yifei Gao, Junhong Ye, Jiaqi Wang, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04370)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly improved the capabilities of web agents. However, effectively navigating complex and dynamic web environments still requires more advanced trajectory-level planning and execution. Prior studies have addressed self-improving agents by collecting extensive GUI trajectories from real-environment interactions. Despite their effectiveness, these approaches encounter two critical challenges: (1) Uncontrollable environment states, where real or sandboxed web environments often yield unstable and non-deterministic feedback, complicating the reproduction and debugging of agent behaviors; and (2) High API costs, as generating even a single interaction trajectory can involve hundreds of queries, leading to considerable API usage and computational expenses. To address these limitations and enable scalable self-improvement for agents, we propose WebSynthesis, a novel framework for trajectory synthesis and training. WebSynthesis leverages a learned world model to simulate virtual web environments, allowing a policy agent to perform efficient and reversible tree-based planning. This approach supports the large-scale generation of diverse and high-quality trajectories, which are subsequently utilized to refine the agent's policy. Experimental results demonstrate that an agent trained using WebSynthesis on a small-scale synthetic dataset achieves performance comparable to or even surpassing that of models trained on large-scale real-world data. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）显著提升了网络代理的能力。然而，有效地在复杂和动态的网络环境中导航仍然需要更高级的轨迹级规划和执行。先前的研究通过收集来自真实环境交互的广泛GUI轨迹来处理自我改进的代理。尽管这些方法有效，但它们遇到了两个关键挑战：（1）不可控的环境状态，其中真实的或沙箱化的网络环境通常会导致不稳定和非确定性的反馈，复杂化了代理行为的重现和调试；（2）高昂的API成本，生成单个交互轨迹可能需要数百次查询，导致大量的API使用和计算开销。为了解决这些限制并使代理的自我改进可扩展，我们提出WebSynthesis，一种新型的轨迹合成与训练框架。WebSynthesis利用学习到的世界模型来模拟虚拟网络环境，使策略代理能够进行高效且可逆的树状规划。这种方法支持大规模生成多样化和高质量的轨迹，随后用于细化代理策略。实验结果表明，使用WebSynthesis在小型合成数据集上训练的代理，在性能上与或甚至超越了在大规模真实世界数据上训练的模型。 

---
# SmartThinker: Learning to Compress and Preserve Reasoning by Step-Level Length Control 

**Title (ZH)**: SmartThinker：通过步骤级长度控制学习压缩与保留推理 

**Authors**: Xingyang He, Xiao Ling, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04348)  

**Abstract**: Large reasoning models (LRMs) have exhibited remarkable reasoning capabilities through inference-time scaling, but this progress has also introduced considerable redundancy and inefficiency into their reasoning processes, resulting in substantial computational waste. Previous work has attempted to mitigate this issue by penalizing the overall length of generated samples during reinforcement learning (RL), with the goal of encouraging a more concise chains of thought. However, we observe that such global length penalty often lead to excessive compression of critical reasoning steps while preserving unnecessary details in simpler ones, yielding a suboptimal trade-off between accuracy and efficiency. To address this issue, we propose SmartThinker, a two-stage learnable framework designed to enable fine-grained control over the length of reasoning chains based on the importance of each individual step. In the first stage, SmartThinker adapts a reasoning model to a short-form reasoning mode through rejection sampling combined with supervised fine-tuning (SFT). In the second stage, SmartThinker applies Step-Level Length Control Policy Optimization (SCPO) to refine the model output distribution, which increases the proportion of length allocated to critical steps while reducing redundancy in less important ones. SCPO consists of four core components: an online importance estimator, a step-level length control reward function, a step-level generalized advantage estimation (S-GAE) and a difficulty-adaptive clipping strategy. Working in concert, these components enable SCPO to implement differentiated length control across reasoning steps. Empirical results across multiple reasoning benchmarks and various backbone models demonstrate that SmartThinker significantly reduces redundant reasoning while achieving comparable or even superior performance to existing methods. 

**Abstract (ZH)**: 大型推理模型通过推理时缩放展示了出色的推理能力，但这一进展也导致其推理过程中的冗余和低效，引发了显著的计算浪费。先前的工作试图通过在强化学习（RL）中惩罚生成样本的总体长度来缓解这一问题，旨在鼓励更加简洁的推理链。然而，我们观察到，这种全局长度惩罚往往会过度压缩关键推理步骤，同时保留不那么关键步骤中的不必要的细节，从而在准确性和效率之间造成了次优权衡。为了解决这一问题，我们提出了一种名为SmartThinker的学习型两阶段框架，该框架旨在根据每一步的重要性对推理链的长度进行细粒度控制。在第一阶段，SmartThinker通过拒绝采样结合监督微调（SFT）将推理模型适应为短形式推理模式。在第二阶段，SmartThinker应用步骤级长度控制策略优化（SCPO）来精炼模型输出分布，增加关键步骤所分配的长度比例，同时减少不那么重要的步骤中的冗余。SCPO包括四个核心组件：在线重要性估计器、步骤级长度控制奖励函数、步骤级泛化优势估计（S-GAE）以及适应难度的裁剪策略。这些组件协同工作，使得SCPO能够在不同的推理步骤中实现差异化的长度控制。在多个推理基准和各种骨干模型上的实证结果表明，SmartThinker在显著减少冗余推理的同时，实现了与现有方法相当甚至更优的性能。 

---
# Voltage Mode Winner-Take-All Circuit for Neuromorphic Systems 

**Title (ZH)**: 基于电压模式的Winner-Take-All电路在类脑系统中的应用 

**Authors**: Abdullah M. Zyarah, Dhireesha Kudithipudi  

**Link**: [PDF](https://arxiv.org/pdf/2507.04338)  

**Abstract**: Recent advances in neuromorphic computing demonstrate on-device learning capabilities with low power consumption. One of the key learning units in these systems is the winner-take-all circuit. In this research, we propose a winner-take-all circuit that can be configured to achieve k-winner and hysteresis properties, simulated in IBM 65 nm node. The circuit dissipated 34.9 $\mu$W of power with a latency of 10.4 ns, while processing 1000 inputs. The utility of the circuit is demonstrated for spatial filtering and classification. 

**Abstract (ZH)**: 近期神经形态计算的进展展示了低功耗下的在器件学习能力，这些系统中的关键学习单元是赢家通吃电路。本研究提出了一种可配置以实现k-赢家和滞回特性的赢家通吃电路，并在IBM 65 nm节点上进行了模拟。该电路在处理1000个输入时消耗了34.9 μW的功率，延时为10.4 ns。电路的应用展示了其在空间滤波和分类中的实用性。 

---
# Answer Set Programming Modulo Theories and Reasoning about Continuous Changes 

**Title (ZH)**: Answer Set Programming Modulo Theories and Continuous Change Reasoning 

**Authors**: Joohyung Lee, Yunsong Meng  

**Link**: [PDF](https://arxiv.org/pdf/2507.04299)  

**Abstract**: Answer Set Programming Modulo Theories (ASPMT) is a new framework of tight integration of answer set programming (ASP) and satisfiability modulo theories (SMT). Similar to the relationship between first-order logic and SMT, it is based on a recent proposal of the functional stable model semantics by fixing interpretations of background theories. Analogously to a known relationship between ASP and SAT, ``tight'' ASPMT programs can be translated into SMT instances. We demonstrate the usefulness of ASPMT by enhancing action language C+ to handle continuous changes as well as discrete changes. We reformulate the semantics of C+ in terms ofASPMT, and show that SMT solvers can be used to compute the language. We also show how the language can represent cumulative effects on continuous resources. 

**Abstract (ZH)**: 基于理论的回答集编程（ASPMT）是一种将回答集编程（ASP）和冲突模理论（SMT）紧密集成的新框架。通过固定背景理论的解释，它基于近期提出的函子稳定模型语义。类似于一阶逻辑与SMT之间的关系，“紧致”的ASPMT程序可以翻译成SMT实例。我们通过增强C+动作语言以处理连续变化和离散变化，展示了ASPMT的应用价值。我们将C+的语言语义重新表述为ASPMT，并展示了如何使用SMT求解器来计算该语言。我们还展示了该语言如何表示连续资源上的累积效果。 

---
# Clustering via Self-Supervised Diffusion 

**Title (ZH)**: 自监督扩散聚类 

**Authors**: Roy Uziel, Irit Chelly, Oren Freifeld, Ari Pakman  

**Link**: [PDF](https://arxiv.org/pdf/2507.04283)  

**Abstract**: Diffusion models, widely recognized for their success in generative tasks, have not yet been applied to clustering. We introduce Clustering via Diffusion (CLUDI), a self-supervised framework that combines the generative power of diffusion models with pre-trained Vision Transformer features to achieve robust and accurate clustering. CLUDI is trained via a teacher-student paradigm: the teacher uses stochastic diffusion-based sampling to produce diverse cluster assignments, which the student refines into stable predictions. This stochasticity acts as a novel data augmentation strategy, enabling CLUDI to uncover intricate structures in high-dimensional data. Extensive evaluations on challenging datasets demonstrate that CLUDI achieves state-of-the-art performance in unsupervised classification, setting new benchmarks in clustering robustness and adaptability to complex data distributions. 

**Abstract (ZH)**: 通过扩散模型进行聚类：CLUDI 

---
# Mpemba Effect in Large-Language Model Training Dynamics: A Minimal Analysis of the Valley-River model 

**Title (ZH)**: 大规模语言模型训练动力学中的Mpemba效应：谷河模型的最小分析 

**Authors**: Sibei Liu, Zhijian Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04206)  

**Abstract**: Learning rate (LR) schedules in large language model (LLM) training often follow empirical templates: warm-up, constant plateau/stable phase, and decay (WSD). However, the mechanistic explanation for this strategy remains underexplored, and the choice of plateau height and decay schedule is largely heuristic. In this paper, we connect training dynamics to a thermodynamic analogy via the Mpemba effect - a phenomenon in which a hotter system cools faster than a colder one when quenched into the same bath. We analyze a class of "valley-river" loss landscapes, where sharp (valley) directions equilibrate quickly, while flatter (river) directions govern global descent. The Mpemba effect provides an explanation for the necessity of the warm-up phase and motivates a high plateau - rather than a low one - for accelerating loss decrease during decay. We show that for certain loss landscapes, there exists an optimal plateau learning rate - the "strong Mpemba point" - at which the slowest mode vanishes, resulting in faster convergence during the decay phase. We derive analytical conditions for its existence and estimate decay dynamics required to preserve the Mpemba advantage. Our minimal model and analysis offer a principled justification for plateau-based schedulers and provide guidance for tuning LR in LLMs with minimal hyperparameter sweep. 

**Abstract (ZH)**: 基于Mpemba效应的大型语言模型训练学习率调度机制研究 

---
# A Technical Survey of Reinforcement Learning Techniques for Large Language Models 

**Title (ZH)**: 大规模语言模型中强化学习技术综述 

**Authors**: Saksham Sahai Srivastava, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2507.04136)  

**Abstract**: Reinforcement Learning (RL) has emerged as a transformative approach for aligning and enhancing Large Language Models (LLMs), addressing critical challenges in instruction following, ethical alignment, and reasoning capabilities. This survey offers a comprehensive foundation on the integration of RL with language models, highlighting prominent algorithms such as Proximal Policy Optimization (PPO), Q-Learning, and Actor-Critic methods. Additionally, it provides an extensive technical overview of RL techniques specifically tailored for LLMs, including foundational methods like Reinforcement Learning from Human Feedback (RLHF) and AI Feedback (RLAIF), as well as advanced strategies such as Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO). We systematically analyze their applications across domains, i.e., from code generation to tool-augmented reasoning. We also present a comparative taxonomy based on reward modeling, feedback mechanisms, and optimization strategies. Our evaluation highlights key trends. RLHF remains dominant for alignment, and outcome-based RL such as RLVR significantly improves stepwise reasoning. However, persistent challenges such as reward hacking, computational costs, and scalable feedback collection underscore the need for continued innovation. We further discuss emerging directions, including hybrid RL algorithms, verifier-guided training, and multi-objective alignment frameworks. This survey serves as a roadmap for researchers advancing RL-driven LLM development, balancing capability enhancement with safety and scalability. 

**Abstract (ZH)**: 强化学习（RL）已成为调整和增强大规模语言模型（LLMs）的一种变革性方法，解决了指令跟随、伦理对齐和推理能力等关键挑战。本文综述了RL与语言模型的整合，重点介绍了诸如 proximal 策略优化（PPO）、Q 学习和演员-评论家方法等 prominent 算法。此外，本文还提供了适用于 LLM 的 RL 技术的全面技术概述，包括基于人类反馈的强化学习（RLHF）和AI反馈（RLAIF）等基础方法，以及直接偏好优化（DPO）和组相对策略优化（GRPO）等先进策略。我们系统地分析了这些方法在不同领域的应用，从代码生成到工具增强的推理。我们还基于奖励建模、反馈机制和优化策略提出了比较分类法。评估表明，RLHF 在对齐方面仍占主导地位，基于结果的 RL（如 RLVR）显著提高了逐步推理能力。然而，奖励作弊、计算成本和可扩展的反馈收集等问题持续存在，需要不断创新。此外，我们讨论了新兴方向，包括混合 RL 算法、验证者引导训练和多目标对齐框架。本文为推进基于 RL 的 LLM 开发的研究人员提供了一份路线图，平衡了能力提升、安全性和可扩展性。 

---
# Enhancing Robustness of LLM-Driven Multi-Agent Systems through Randomized Smoothing 

**Title (ZH)**: 通过随机化平滑增强基于LLM的多智能体系统的鲁棒性 

**Authors**: Jinwei Hu, Yi Dong, Zhengtao Ding, Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04105)  

**Abstract**: This paper presents a defense framework for enhancing the safety of large language model (LLM) empowered multi-agent systems (MAS) in safety-critical domains such as aerospace. We apply randomized smoothing, a statistical robustness certification technique, to the MAS consensus context, enabling probabilistic guarantees on agent decisions under adversarial influence. Unlike traditional verification methods, our approach operates in black-box settings and employs a two-stage adaptive sampling mechanism to balance robustness and computational efficiency. Simulation results demonstrate that our method effectively prevents the propagation of adversarial behaviors and hallucinations while maintaining consensus performance. This work provides a practical and scalable path toward safe deployment of LLM-based MAS in real-world, high-stakes environments. 

**Abstract (ZH)**: 本文提出了一种防御框架，旨在增强大型语言模型（LLM）驱动的多agent系统（MAS）在航空航天等安全关键领域中的安全性。我们将在MAS共识语境中应用随机化光滑技术，这是一种统计鲁棒性验证技术，能够在对抗性影响下为代理决策提供概率保证。与传统验证方法不同，我们的方法在黑盒设置中运行，并采用两阶段自适应采样机制以平衡鲁棒性和计算效率。仿真结果表明，我们的方法有效防止了对抗行为和幻觉的传播，同时保持了共识性能。本文为在实际高风险环境中安全部署基于LLM的MAS提供了实用且可扩展的途径。 

---
# How to Train Your LLM Web Agent: A Statistical Diagnosis 

**Title (ZH)**: 如何训练你的LLM网络代理：一种统计诊断 

**Authors**: Dheeraj Vattikonda, Santhoshi Ravichandran, Emiliano Penaloza, Hadi Nekoei, Megh Thakkar, Thibault Le Sellier de Chezelles, Nicolas Gontier, Miguel Muñoz-Mármol, Sahar Omidi Shayegan, Stefania Raimondo, Xue Liu, Alexandre Drouin, Laurent Charlin, Alexandre Piché, Alexandre Lacoste, Massimo Caccia  

**Link**: [PDF](https://arxiv.org/pdf/2507.04103)  

**Abstract**: LLM-based web agents have recently made significant progress, but much of it has occurred in closed-source systems, widening the gap with open-source alternatives. Progress has been held back by two key challenges: first, a narrow focus on single-step tasks that overlooks the complexity of multi-step web interactions; and second, the high compute costs required to post-train LLM-based web agents. To address this, we present the first statistically grounded study on compute allocation for LLM web-agent post-training. Our approach uses a two-stage pipeline, training a Llama 3.1 8B student to imitate a Llama 3.3 70B teacher via supervised fine-tuning (SFT), followed by on-policy reinforcement learning. We find this process highly sensitive to hyperparameter choices, making exhaustive sweeps impractical. To spare others from expensive trial-and-error, we sample 1,370 configurations and use bootstrapping to estimate effective hyperparameters. Our results show that combining SFT with on-policy RL consistently outperforms either approach alone on both WorkArena and MiniWob++. Further, this strategy requires only 55% of the compute to match the peak performance of pure SFT on MiniWob++, effectively pushing the compute-performance Pareto frontier, and is the only strategy that can close the gap with closed-source models. 

**Abstract (ZH)**: 基于LLM的网络代理最近取得了显著进展，但大多发生在闭源系统中，与开源替代方案之间差距加大。进展受限于两大关键挑战：首先，过度关注单步骤任务，忽视了多步骤网络交互的复杂性；其次，后训练LLM网络代理所需的高计算成本。为应对这一问题，我们提出了首个基于统计依据的实验研究，探讨后训练LLM网络代理的计算资源分配。我们采用两阶段管道，通过监督微调（SFT）训练一个Llama 3.1 8B学生去模仿Llama 3.3 70B教师，随后利用策略性强化学习。我们发现该过程对超参数选择高度敏感，使得全面调整不可行。为避免他人经历昂贵的试错过程，我们采样了1,370种配置并利用自助法估计有效的超参数。结果显示，将SFT与策略性强化学习相结合在WorkArena和MiniWob++上的一致性上优于两种方法单独使用的效果。此外，这一策略仅需55%的计算资源即可达到纯SFT在MiniWob++上的最佳性能，有效地推动了计算-性能帕累托前沿，并且是唯一能缩小与闭源模型差距的策略。 

---
# HAWK: A Hierarchical Workflow Framework for Multi-Agent Collaboration 

**Title (ZH)**: HAWK：多智能体协作的层次化工作流框架 

**Authors**: Yuyang Cheng, Yumiao Xu, Chaojia Yu, Yong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.04067)  

**Abstract**: Contemporary multi-agent systems encounter persistent challenges in cross-platform interoperability, dynamic task scheduling, and efficient resource sharing. Agents with heterogeneous implementations often lack standardized interfaces; collaboration frameworks remain brittle and hard to extend; scheduling policies are static; and inter-agent state synchronization is insufficient. We propose Hierarchical Agent Workflow (HAWK), a modular framework comprising five layers-User, Workflow, Operator, Agent, and Resource-and supported by sixteen standardized interfaces. HAWK delivers an end-to-end pipeline covering task parsing, workflow orchestration, intelligent scheduling, resource invocation, and data synchronization. At its core lies an adaptive scheduling and optimization module in the Workflow Layer, which harnesses real-time feedback and dynamic strategy adjustment to maximize utilization. The Resource Layer provides a unified abstraction over heterogeneous data sources, large models, physical devices, and third-party services&tools, simplifying cross-domain information retrieval. We demonstrate HAWK's scalability and effectiveness via CreAgentive, a multi-agent novel-generation prototype, which achieves marked gains in throughput, lowers invocation complexity, and improves system controllability. We also show how hybrid deployments of large language models integrate seamlessly within HAWK, highlighting its flexibility. Finally, we outline future research avenues-hallucination mitigation, real-time performance tuning, and enhanced cross-domain adaptability-and survey prospective applications in healthcare, government, finance, and education. 

**Abstract (ZH)**: 当代多-agent系统在跨平台互操作性、动态任务调度和高效资源分享方面面临持续挑战。不同实现的代理缺少标准化接口；协作框架脆弱且难以扩展；调度策略静态且不够智能；代理间状态同步不足。我们提出了一种分层代理工作流（HAWK）框架，该框架包含五层-用户、工作流、操作员、代理和资源，并由十六个标准化接口支持。HAWK提供了一整套涵盖任务解析、工作流编排、智能调度、资源调用和数据同步的端到端管道。其核心是在工作流层中集成了一个自适应调度和优化模块，该模块利用实时反馈和动态策略调整来优化资源利用。资源层提供了一种统一的异构数据源、大规模模型、物理设备和第三方服务及工具的抽象，简化了跨域信息检索。通过CreAgentive多-agent新型生成原型，我们展示了HAWK的可扩展性和有效性，实现了吞吐量显著提升、调用复杂度降低和系统可控性的提高。我们还展示了大型语言模型在HAWK中的无缝集成，突显了其灵活性。最后，我们提出了未来研究方向——幻觉抑制、实时性能调优和增强的跨域适应性，并概述了其在医疗、政府、金融和教育等领域的潜在应用。 

---
# Ready Jurist One: Benchmarking Language Agents for Legal Intelligence in Dynamic Environments 

**Title (ZH)**: Ready Jurist One: 动态环境中文书代理的法律智能基准测试 

**Authors**: Zheng Jia, Shengbin Yue, Wei Chen, Siyuan Wang, Yidong Liu, Yun Song, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.04037)  

**Abstract**: The gap between static benchmarks and the dynamic nature of real-world legal practice poses a key barrier to advancing legal intelligence. To this end, we introduce J1-ENVS, the first interactive and dynamic legal environment tailored for LLM-based agents. Guided by legal experts, it comprises six representative scenarios from Chinese legal practices across three levels of environmental complexity. We further introduce J1-EVAL, a fine-grained evaluation framework, designed to assess both task performance and procedural compliance across varying levels of legal proficiency. Extensive experiments on 17 LLM agents reveal that, while many models demonstrate solid legal knowledge, they struggle with procedural execution in dynamic settings. Even the SOTA model, GPT-4o, falls short of 60% overall performance. These findings highlight persistent challenges in achieving dynamic legal intelligence and offer valuable insights to guide future research. 

**Abstract (ZH)**: 静态基准与现实法律实践动态特性之间的差距是推动法律智能发展的一大障碍。为此，我们引入J1-ENVS，这是首个专为基于LLM的代理设计的交互式和动态法律环境。该环境在法律专家的指导下，包含了来自中国法律实践的六个代表性场景，涉及不同复杂度的环境层次。我们还引入了J1-EVAL，这是一种精细的评估框架，旨在在不同法律熟练程度的层次上评估任务性能和程序合规性。对17个LLM代理的广泛实验表明，虽然许多模型展示了扎实的法律知识，但在动态环境中执行程序方面存在困难。即使是当前最先进的模型GPT-4o的整体性能也未达到60%。这些发现强调了在实现动态法律智能方面持续存在的挑战，并为未来研究提供了宝贵见解。 

---
# Lyria: A General LLM-Driven Genetic Algorithm Framework for Problem Solving 

**Title (ZH)**: Lyria：一个通用的大语言模型驱动的遗传算法框架用于问题求解 

**Authors**: Weizhi Tang, Kwabena Nuamah, Vaishak Belle  

**Link**: [PDF](https://arxiv.org/pdf/2507.04034)  

**Abstract**: While Large Language Models (LLMs) have demonstrated impressive abilities across various domains, they still struggle with complex problems characterized by multi-objective optimization, precise constraint satisfaction, immense solution spaces, etc. To address the limitation, drawing on the superior semantic understanding ability of LLMs and also the outstanding global search and optimization capability of genetic algorithms, we propose to capitalize on their respective strengths and introduce Lyria, a general LLM-driven genetic algorithm framework, comprising 7 essential components. Through conducting extensive experiments with 4 LLMs across 3 types of problems, we demonstrated the efficacy of Lyria. Additionally, with 7 additional ablation experiments, we further systematically analyzed and elucidated the factors that affect its performance. 

**Abstract (ZH)**: 尽管大规模语言模型（LLMs）在各个领域展现了出色的性能，但在多目标优化、精确约束满足、庞大的解空间等复杂问题上仍然存在局限性。为解决这一问题，我们利用LLMs在语义理解上的优势以及遗传算法在全球搜索和优化方面的卓越能力，提出了一种充分利用两者优势的框架——Lyria，该框架包含7个核心组件。通过在3类问题上使用4种LLM进行广泛实验，我们证明了Lyria的有效性。此外，通过7项额外的消融实验，我们系统地分析并阐明了影响其性能的因素。 

---
# Toward Better Generalisation in Uncertainty Estimators: Leveraging Data-Agnostic Features 

**Title (ZH)**: 面向不确定性估计中的更好泛化：利用数据无关特征 

**Authors**: Thuy An Ha, Bao Quoc Vo  

**Link**: [PDF](https://arxiv.org/pdf/2507.03998)  

**Abstract**: Large Language Models (LLMs) often generate responses that are factually incorrect yet expressed with high confidence, which can pose serious risks for end users. To address this, it is essential for LLMs not only to produce answers but also to provide accurate estimates of their correctness. Uncertainty quantification methods have been introduced to assess the quality of LLM outputs, with factual accuracy being a key aspect of that quality. Among these methods, those that leverage hidden states to train probes have shown particular promise, as these internal representations encode information relevant to the factuality of responses, making this approach the focus of this paper. However, the probe trained on the hidden states of one dataset often struggles to generalise to another dataset of a different task or domain. To address this limitation, we explore combining data-agnostic features with hidden-state features and assess whether this hybrid feature set enhances out-of-domain performance. We further examine whether selecting only the most informative hidden-state features, thereby discarding task-specific noise, enables the data-agnostic features to contribute more effectively. The experiment results indicate that although introducing data-agnostic features generally enhances generalisation performance in most cases, in certain scenarios their inclusion degrades performance. A similar pattern emerges when retaining only the most important hidden-state features - adding data-agnostic features does not consistently further enhance performance compared to using the full set of hidden-state features. A closer analysis reveals that, in some specific cases, the trained probe underweights the data-agnostic features relative to the hidden-state features, which we believe is the main reason why the results are inconclusive. 

**Abstract (ZH)**: 大型语言模型（LLMs）常常生成事实性错误但表达高度自信的回应，这对终端用户构成了严重风险。为此，LLMs不仅需要提供答案，还需要提供其正确性的准确估计。已引入不确定性量化方法来评估LLM输出的质量，事实准确性是这一质量的关键方面。其中，利用隐藏状态训练探针的方法显示出特别的前景，因为这些内部表示包含了与回应事实性相关的信息，因此将这种方法作为本文重点。然而，针对一个数据集训练的探针往往难以迁移到不同任务或领域的新数据集上。为解决这一局限，我们探讨了结合数据无关特征与隐藏状态特征的可行性，并评估这种混合特征集是否能改善域外性能。我们还研究了仅选择最具信息性的隐藏状态特征，从而丢弃任务特定噪声，是否能更有效地使数据无关特征发挥作用。实验结果表明，虽然在大多数情况下引入数据无关特征一般能提升泛化性能，但在某些场景下其加入反而会降低性能。当仅保留最重要的隐藏状态特征时，同样观察到添加数据无关特征并不总能比使用全部隐藏状态特征进一步提升性能。进一步分析发现，在某些特定情况下，训练探针对数据无关特征的权重相对于隐藏状态特征较轻，我们认为这是导致结果不明确的主要原因。 

---
# An ASP-Based Framework for MUSes 

**Title (ZH)**: 基于ASP的MUSes框架 

**Authors**: Mohimenul Kabir, Kuldeep S Meel  

**Link**: [PDF](https://arxiv.org/pdf/2507.03929)  

**Abstract**: Given an unsatisfiable formula, understanding the core reason for unsatisfiability is crucial in several applications. One effective way to capture this is through the minimal unsatisfiable subset (MUS), the subset-minimal set of clauses that remains unsatisfiable. Current research broadly focuses on two directions: (i) enumerating as many MUSes as possible within a given time limit, and (ii) counting the total number of MUSes for a given unsatisfiable formula.
In this paper, we introduce an answer set programming-based framework, named MUS-ASP, designed for online enumeration of MUSes. ASP is a powerful tool for its strengths in knowledge representation and is particularly suitable for specifying complex combinatorial problems. By translating MUS enumeration into answer set solving, MUS-ASP leverages the computational efficiency of state-of-the-art ASP systems. Our extensive experimental evaluation demonstrates the effectiveness of MUS-ASP and highlights the acceleration in both MUS enumeration and counting tasks, particularly when integrated within hybrid solvers, including the framework proposed in this paper. 

**Abstract (ZH)**: 基于ASP的MUS在线枚举框架：MUS-ASP 

---
# CortexDebate: Debating Sparsely and Equally for Multi-Agent Debate 

**Title (ZH)**: CortexDebate: 多代理辩论中的稀疏平等争论 

**Authors**: Yiliu Sun, Zicheng Zhao, Sheng Wan, Chen Gong  

**Link**: [PDF](https://arxiv.org/pdf/2507.03928)  

**Abstract**: Nowadays, single Large Language Model (LLM) struggles with critical issues such as hallucination and inadequate reasoning abilities. To mitigate these issues, Multi-Agent Debate (MAD) has emerged as an effective strategy, where LLM agents engage in in-depth debates with others on tasks. However, existing MAD methods face two major issues: (a) too lengthy input contexts, which causes LLM agents to get lost in plenty of input information and experiences performance drop; and (b) the overconfidence dilemma, where self-assured LLM agents dominate the debate, leading to low debating effectiveness. To address these limitations, we propose a novel MAD method called "CortexDebate". Inspired by the human brain's tendency to establish a sparse and dynamically optimized network among cortical areas governed by white matter, CortexDebate constructs a sparse debating graph among LLM agents, where each LLM agent only debates with the ones that are helpful to it. To optimize the graph, we propose a module named McKinsey-based Debate Matter (MDM), which acts as an artificial analog to white matter. By integrating the McKinsey Trust Formula, a well-established measure of trustworthiness from sociology, MDM enables credible evaluations that guide graph optimization. The effectiveness of our CortexDebate has been well demonstrated by extensive experimental results across eight datasets from four task types. 

**Abstract (ZH)**: CortexDebate：借鉴大脑皮层间稀疏优化网络的多智能体辩论新方法 

---
# Animation Needs Attention: A Holistic Approach to Slides Animation Comprehension with Visual-Language Models 

**Title (ZH)**: 动画需要关注：基于视觉-语言模型的整体幻灯片动画理解方法 

**Authors**: Yifan Jiang, Yibo Xue, Yukun Kang, Pin Zheng, Jian Peng, Feiran Wu, Changliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.03916)  

**Abstract**: Slide animations, such as fade-ins, fly-ins, and wipes, are critical for audience engagement, efficient information delivery, and vivid visual expression. However, most AI-driven slide-generation tools still lack native animation support, and existing vision-language models (VLMs) struggle with animation tasks due to the absence of public datasets and limited temporal-reasoning capabilities. To address this gap, we release the first public dataset for slide-animation modeling: 12,000 triplets of natural-language descriptions, animation JSON files, and rendered videos, collectively covering every built-in PowerPoint effect. Using this resource, we fine-tune Qwen-2.5-VL-7B with Low-Rank Adaptation (LoRA) and achieve consistent improvements over GPT-4.1 and Gemini-2.5-Pro in BLEU-4, ROUGE-L, SPICE, and our Coverage-Order-Detail Assessment (CODA) metric, which evaluates action coverage, temporal order, and detail fidelity. On a manually curated test set of slides, the LoRA model increases BLEU-4 by around 60%, ROUGE-L by 30%, and shows significant improvements in CODA-detail. This demonstrates that low-rank adaptation enables reliable temporal reasoning and generalization beyond synthetic data. Overall, our dataset, LoRA-enhanced model, and CODA metric provide a rigorous benchmark and foundation for future research on VLM-based dynamic slide generation. 

**Abstract (ZH)**: 幻灯片动画（如渐显、飞入和擦除）对于增强观众参与度、高效信息传递和生动视觉表达至关重要。然而，大多数基于AI的幻灯片生成工具仍缺乏内置动画支持，现有的视觉-语言模型（VLM）由于缺乏公开数据集和有限的时序推理能力，在动画任务上的表现受限。为解决这一问题，我们发布了首个公开的幻灯片动画建模数据集：包含12,000组自然语言描述、动画JSON文件和渲染视频的三元组，覆盖了PowerPoint的所有内置效果。利用这一资源，我们使用低秩适应（LoRA）对Qwen-2.5-VL-7B进行微调，并在BLEU-4、ROUGE-L、SPICE和我们的Coverage-Order-Detail Assessment（CODA）指标上实现了对GPT-4.1和Gemini-2.5-Pro的一致改进，其中CODA指标评估动作覆盖、时间顺序和细节保真度。在手动整理的一组测试幻灯片上，LoRA模型的BLEU-4提高了约60%，ROUGE-L提高了30%，并在CODA细节方面显示出显著改进。这表明低秩适应能够实现可靠的时序推理，并能在合成数据之外进行泛化。总体而言，我们的数据集、LoRA增强模型和CODA指标为基于VLM的动态幻灯片生成的未来研究提供了严格的基准和基础。 

---
# Agent Exchange: Shaping the Future of AI Agent Economics 

**Title (ZH)**: 智能代理交换：塑造未来智能代理经济 

**Authors**: Yingxuan Yang, Ying Wen, Jun Wang, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03904)  

**Abstract**: The rise of Large Language Models (LLMs) has transformed AI agents from passive computational tools into autonomous economic actors. This shift marks the emergence of the agent-centric economy, in which agents take on active economic roles-exchanging value, making strategic decisions, and coordinating actions with minimal human oversight. To realize this vision, we propose Agent Exchange (AEX), a specialized auction platform designed to support the dynamics of the AI agent marketplace. AEX offers an optimized infrastructure for agent coordination and economic participation. Inspired by Real-Time Bidding (RTB) systems in online advertising, AEX serves as the central auction engine, facilitating interactions among four ecosystem components: the User-Side Platform (USP), which translates human goals into agent-executable tasks; the Agent-Side Platform (ASP), responsible for capability representation, performance tracking, and optimization; Agent Hubs, which coordinate agent teams and participate in AEX-hosted auctions; and the Data Management Platform (DMP), ensuring secure knowledge sharing and fair value attribution. We outline the design principles and system architecture of AEX, laying the groundwork for agent-based economic infrastructure in future AI ecosystems. 

**Abstract (ZH)**: 大型语言模型(Large Language Models)的兴起已将AI代理从被动的计算工具转变为自主的经济行为者。这一转变标志着代理中心经济的 emergence，在这种经济中，代理承担起积极的经济角色——交换价值、作出战略决策并以最少的人为监督协调行动。为了实现这一愿景，我们提出Agent Exchange (AEX) ——一个专门的拍卖平台，旨在支持AI代理市场的动态。AEX 提供了代理协调和经济参与的优化基础设施。受到在线广告中实时竞价系统（Real-Time Bidding, RTB）的启发，AEX 作为中心拍卖引擎，促进用户侧平台（User-Side Platform, USP）、代理侧平台（Agent-Side Platform, ASP）、代理枢纽（Agent Hubs）和数据管理平台（Data Management Platform, DMP）这四个生态系统组件之间的互动。AEX 确保知识的安全共享和公平的价值归因。我们概述了 AEX 的设计原则和系统架构，为未来AI生态系统的基于代理的经济基础设施奠定基础。 

---
# LLMs model how humans induce logically structured rules 

**Title (ZH)**: LLMs模型人类如何诱导逻辑结构化的规则 

**Authors**: Alyssa Loo, Ellie Pavlick, Roman Feiman  

**Link**: [PDF](https://arxiv.org/pdf/2507.03876)  

**Abstract**: A central goal of cognitive science is to provide a computationally explicit account of both the structure of the mind and its development: what are the primitive representational building blocks of cognition, what are the rules via which those primitives combine, and where do these primitives and rules come from in the first place? A long-standing debate concerns the adequacy of artificial neural networks as computational models that can answer these questions, in particular in domains related to abstract cognitive function, such as language and logic. This paper argues that recent advances in neural networks -- specifically, the advent of large language models (LLMs) -- represent an important shift in this debate. We test a variety of LLMs on an existing experimental paradigm used for studying the induction of rules formulated over logical concepts. Across four experiments, we find converging empirical evidence that LLMs provide at least as good a fit to human behavior as models that implement a Bayesian probablistic language of thought (pLoT), which have been the best computational models of human behavior on the same task. Moreover, we show that the LLMs make qualitatively different predictions about the nature of the rules that are inferred and deployed in order to complete the task, indicating that the LLM is unlikely to be a mere implementation of the pLoT solution. Based on these results, we argue that LLMs may instantiate a novel theoretical account of the primitive representations and computations necessary to explain human logical concepts, with which future work in cognitive science should engage. 

**Abstract (ZH)**: 认知科学的一个核心目标是提供一种计算上明确的认知结构及其发展账户：认知的基本表征构建块是什么，这些构建块是如何组合的规则是什么，这些构建块和规则最初是如何产生的？长期以来，关于人工神经网络作为能够回答这些问题的计算模型的适当性存在争议，特别是在涉及抽象认知功能的领域，如语言和逻辑。本文认为，神经网络的近期进展——特别是大型语言模型（LLMs）的出现——在这一争议中代表了一个重要转折。我们测试了几种LLMs在研究基于逻辑概念规则归纳的现有实验范式中的表现。在四次实验中，我们发现LLMs至少与实施贝叶斯概率语言（pLoT）模型的表现相当，后者在相同任务中是人类行为的最佳计算模型。此外，我们证明了LLMs对任务中推断和执行的规则的性质提出了不同的预测，表明LLM不太可能是pLoT解决方案的简单实现。基于这些结果，我们认为LLMs可能实现了一种新的理论账户，以解释人类逻辑概念所需的原始表征和计算，未来的认知科学研究应关注这一点。 

---
# Uncovering Systemic and Environment Errors in Autonomous Systems Using Differential Testing 

**Title (ZH)**: 使用差异测试揭露自主系统中的系统级和环境错误 

**Authors**: Rahil P Mehta, Yashwanthi Anand, Manish Motwani, Sandhya Saisubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2507.03870)  

**Abstract**: When an autonomous agent behaves undesirably, including failure to complete a task, it can be difficult to determine whether the behavior is due to a systemic agent error, such as flaws in the model or policy, or an environment error, where a task is inherently infeasible under a given environment configuration, even for an ideal agent. As agents and their environments grow more complex, identifying the error source becomes increasingly difficult but critical for reliable deployment. We introduce AIProbe, a novel black-box testing technique that applies differential testing to attribute undesirable agent behaviors either to agent deficiencies, such as modeling or training flaws, or due to environmental infeasibility. AIProbe first generates diverse environmental configurations and tasks for testing the agent, by modifying configurable parameters using Latin Hypercube sampling. It then solves each generated task using a search-based planner, independent of the agent. By comparing the agent's performance to the planner's solution, AIProbe identifies whether failures are due to errors in the agent's model or policy, or due to unsolvable task conditions. Our evaluation across multiple domains shows that AIProbe significantly outperforms state-of-the-art techniques in detecting both total and unique errors, thereby contributing to a reliable deployment of autonomous agents. 

**Abstract (ZH)**: 当自主代理行为不当，包括未能完成任务时，确定行为是由代理系统的错误（如模型或策略中的缺陷）还是环境错误（在给定环境配置下任务本质上不可行）引起变得困难，尤其是在代理及其环境变得更加复杂的情况下，识别错误源变得越来越困难但至关重要，以确保可靠的部署。我们引入了AIProbe，这是一种新颖的黑盒测试技术，通过差异性测试来归因代理的不当行为是由于代理缺陷（如建模或训练缺陷）还是环境不可行性。AIProbe 首先通过使用拉丁超立方抽样修改可配置参数来生成多样化的环境配置和任务，用于测试代理。然后，它使用基于搜索的计划器独立解决每个生成的任务。通过将代理的表现与计划器的解决方案进行比较，AIProbe 确定失败是由于代理模型或策略中的错误还是由于无法解决的任务条件。我们在多个领域的评估表明，AIProbe 在检测总体和独特错误方面明显优于最先进的技术，从而有助于自主代理的可靠部署。 

---
# From Query to Explanation: Uni-RAG for Multi-Modal Retrieval-Augmented Learning in STEM 

**Title (ZH)**: 从查询到解释：Uni-RAG在STEM领域多模态检索增强学习中的应用 

**Authors**: Xinyi Wu, Yanhao Jia, Luwei Xiao, Shuai Zhao, Fengkuang Chiang, Erik Cambria  

**Link**: [PDF](https://arxiv.org/pdf/2507.03868)  

**Abstract**: In AI-facilitated teaching, leveraging various query styles to interpret abstract educational content is crucial for delivering effective and accessible learning experiences. However, existing retrieval systems predominantly focus on natural text-image matching and lack the capacity to address the diversity and ambiguity inherent in real-world educational scenarios. To address this limitation, we develop a lightweight and efficient multi-modal retrieval module, named Uni-Retrieval, which extracts query-style prototypes and dynamically matches them with tokens from a continually updated Prompt Bank. This Prompt Bank encodes and stores domain-specific knowledge by leveraging a Mixture-of-Expert Low-Rank Adaptation (MoE-LoRA) module and can be adapted to enhance Uni-Retrieval's capability to accommodate unseen query types at test time. To enable natural language educational content generation, we integrate the original Uni-Retrieval with a compact instruction-tuned language model, forming a complete retrieval-augmented generation pipeline named Uni-RAG. Given a style-conditioned query, Uni-RAG first retrieves relevant educational materials and then generates human-readable explanations, feedback, or instructional content aligned with the learning objective. Experimental results on SER and other multi-modal benchmarks show that Uni-RAG outperforms baseline retrieval and RAG systems in both retrieval accuracy and generation quality, while maintaining low computational cost. Our framework provides a scalable, pedagogically grounded solution for intelligent educational systems, bridging retrieval and generation to support personalized, explainable, and efficient learning assistance across diverse STEM scenarios. 

**Abstract (ZH)**: 基于AI辅助教学中的多模态查询风格检索模块Uni-Retrieval及其应用 

---
# Participatory Evolution of Artificial Life Systems via Semantic Feedback 

**Title (ZH)**: 通过语义反馈的人工生命系统参与式演化 

**Authors**: Shuowen Li, Kexin Wang, Minglu Fang, Danqi Huang, Ali Asadipour, Haipeng Mi, Yitong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.03839)  

**Abstract**: We present a semantic feedback framework that enables natural language to guide the evolution of artificial life systems. Integrating a prompt-to-parameter encoder, a CMA-ES optimizer, and CLIP-based evaluation, the system allows user intent to modulate both visual outcomes and underlying behavioral rules. Implemented in an interactive ecosystem simulation, the framework supports prompt refinement, multi-agent interaction, and emergent rule synthesis. User studies show improved semantic alignment over manual tuning and demonstrate the system's potential as a platform for participatory generative design and open-ended evolution. 

**Abstract (ZH)**: 一种基于语义反馈的自然语言引导人工生命系统演化框架 

---
# Economic Evaluation of LLMs 

**Title (ZH)**: LLMs的经济效益评估 

**Authors**: Michael J. Zellinger, Matt Thomson  

**Link**: [PDF](https://arxiv.org/pdf/2507.03834)  

**Abstract**: Practitioners often navigate LLM performance trade-offs by plotting Pareto frontiers of optimal accuracy-cost trade-offs. However, this approach offers no way to compare between LLMs with distinct strengths and weaknesses: for example, a cheap, error-prone model vs a pricey but accurate one. To address this gap, we propose economic evaluation of LLMs. Our framework quantifies the performance trade-off of an LLM as a single number based on the economic constraints of a concrete use case, all expressed in dollars: the cost of making a mistake, the cost of incremental latency, and the cost of abstaining from a query. We apply our economic evaluation framework to compare the performance of reasoning and non-reasoning models on difficult questions from the MATH benchmark, discovering that reasoning models offer better accuracy-cost tradeoffs as soon as the economic cost of a mistake exceeds \$0.01. In addition, we find that single large LLMs often outperform cascades when the cost of making a mistake is as low as \$0.1. Overall, our findings suggest that when automating meaningful human tasks with AI models, practitioners should typically use the most powerful available model, rather than attempt to minimize AI deployment costs, since deployment costs are likely dwarfed by the economic impact of AI errors. 

**Abstract (ZH)**: 实践者经常通过绘制最优准确度-成本trade-off的Pareto前沿来权衡LLM的性能。然而，这种方法无法比较具有不同强项和弱项的LLM：例如，便宜但出错几率高的模型与昂贵但准确的模型。为了填补这一空白，我们提出对LLM进行经济评估。我们的框架根据具体应用场景中的经济约束，将LLM的性能trade-off量化为一个数值，单位均为美元：错误的成本、增益延迟的成本以及拒绝查询的成本。我们应用经济评估框架比较了推理和非推理模型在MATH基准中难以回答的问题上的性能，发现只要错误的经济成本超过0.01美元，推理模型就提供了更好的准确度-成本trade-off。此外，我们发现当错误的成本低至0.1美元时，单个大型LLM通常优于级联模型。总体而言，我们的研究结果表明，在使用AI模型自动化有意义的人类任务时，实践者通常应使用最强大的可用模型，而不是试图最小化AI部署成本，因为部署成本很可能远小于AI错误的经济影响。 

---
# RELRaE: LLM-Based Relationship Extraction, Labelling, Refinement, and Evaluation 

**Title (ZH)**: RELRaE: 基于大语言模型的关系提取、标注、修正与评估 

**Authors**: George Hannah, Jacopo de Berardinis, Terry R. Payne, Valentina Tamma, Andrew Mitchell, Ellen Piercy, Ewan Johnson, Andrew Ng, Harry Rostron, Boris Konev  

**Link**: [PDF](https://arxiv.org/pdf/2507.03829)  

**Abstract**: A large volume of XML data is produced in experiments carried out by robots in laboratories. In order to support the interoperability of data between labs, there is a motivation to translate the XML data into a knowledge graph. A key stage of this process is the enrichment of the XML schema to lay the foundation of an ontology schema. To achieve this, we present the RELRaE framework, a framework that employs large language models in different stages to extract and accurately label the relationships implicitly present in the XML schema. We investigate the capability of LLMs to accurately generate these labels and then evaluate them. Our work demonstrates that LLMs can be effectively used to support the generation of relationship labels in the context of lab automation, and that they can play a valuable role within semi-automatic ontology generation frameworks more generally. 

**Abstract (ZH)**: 实验室中由机器人实验产生的大量XML数据需要转化为知识图谱以支持实验室间的数据互操作性。这一过程的关键阶段是丰富XML模式，以构建本体模式的基础。为此，我们提出了RELRaE框架，该框架在不同阶段使用大型语言模型来提取并准确标注XML模式中隐含的关系。我们研究了大型语言模型生成这些标签的准确度，并对其进行评估。我们的工作证明了大型语言模型可以在实验室自动化背景下有效支持关系标签的生成，并且可以在更广泛的半自动本体生成框架中发挥重要作用。 

---
# Leveraging Large Language Models for Tacit Knowledge Discovery in Organizational Contexts 

**Title (ZH)**: 利用大型语言模型在组织情境中发现隐性知识 

**Authors**: Gianlucca Zuin, Saulo Mastelini, Túlio Loures, Adriano Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2507.03811)  

**Abstract**: Documenting tacit knowledge in organizations can be a challenging task due to incomplete initial information, difficulty in identifying knowledgeable individuals, the interplay of formal hierarchies and informal networks, and the need to ask the right questions. To address this, we propose an agent-based framework leveraging large language models (LLMs) to iteratively reconstruct dataset descriptions through interactions with employees. Modeling knowledge dissemination as a Susceptible-Infectious (SI) process with waning infectivity, we conduct 864 simulations across various synthetic company structures and different dissemination parameters. Our results show that the agent achieves 94.9% full-knowledge recall, with self-critical feedback scores strongly correlating with external literature critic scores. We analyze how each simulation parameter affects the knowledge retrieval process for the agent. In particular, we find that our approach is able to recover information without needing to access directly the only domain specialist. These findings highlight the agent's ability to navigate organizational complexity and capture fragmented knowledge that would otherwise remain inaccessible. 

**Abstract (ZH)**: 利用大型语言模型基于代理的框架在组织中记录隐性知识 

---
# Generating Novelty in Open-World Multi-Agent Strategic Board Games 

**Title (ZH)**: 开放世界多agent战略棋盘游戏中的新颖性生成 

**Authors**: Mayank Kejriwal, Shilpa Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2507.03802)  

**Abstract**: We describe GNOME (Generating Novelty in Open-world Multi-agent Environments), an experimental platform that is designed to test the effectiveness of multi-agent AI systems when faced with \emph{novelty}. GNOME separates the development of AI gameplaying agents with the simulator, allowing \emph{unanticipated} novelty (in essence, novelty that is not subject to model-selection bias). Using a Web GUI, GNOME was recently demonstrated at NeurIPS 2020 using the game of Monopoly to foster an open discussion on AI robustness and the nature of novelty in real-world environments. In this article, we further detail the key elements of the demonstration, and also provide an overview of the experimental design that is being currently used in the DARPA Science of Artificial Intelligence and Learning for Open-World Novelty (SAIL-ON) program to evaluate external teams developing novelty-adaptive gameplaying agents. 

**Abstract (ZH)**: 我们描述了GNOME（生成开放世界多智能体环境中的新颖性），一个实验平台，旨在测试多智能体AI系统在面对新颖性时的有效性。GNOME通过将AI游戏代理的开发与模拟器分离，实现了未预见的新颖性（本质上，不受模型选择偏差影响的新颖性）。通过网络GUI，GNOME在2020年NeurIPS会议上使用垄断游戏进行演示，旨在促进AI鲁棒性以及现实环境中新颖性的本质的公开讨论。在本文中，我们进一步详细介绍了演示的关键要素，并提供了一种实验设计的概述，该设计目前正在DARPA人工智能与学习科学为开放世界新颖性（SAIL-ON）计划中用于评估开发适应新颖性的游戏代理的外部团队。 

---
# Learning Dark Souls Combat Through Pixel Input With Neuroevolution 

**Title (ZH)**: 通过像素输入的神经进化学习《暗魂》战斗 

**Authors**: Jim O'Connor, Gary B. Parker, Mustafa Bugti  

**Link**: [PDF](https://arxiv.org/pdf/2507.03793)  

**Abstract**: This paper investigates the application of Neuroevolution of Augmenting Topologies (NEAT) to automate gameplay in Dark Souls, a notoriously challenging action role-playing game characterized by complex combat mechanics, dynamic environments, and high-dimensional visual inputs. Unlike traditional reinforcement learning or game playing approaches, our method evolves neural networks directly from raw pixel data, circumventing the need for explicit game-state information. To facilitate this approach, we introduce the Dark Souls API (DSAPI), a novel Python framework leveraging real-time computer vision techniques for extracting critical game metrics, including player and enemy health states. Using NEAT, agents evolve effective combat strategies for defeating the Asylum Demon, the game's initial boss, without predefined behaviors or domain-specific heuristics. Experimental results demonstrate that evolved agents achieve up to a 35% success rate, indicating the viability of neuroevolution in addressing complex, visually intricate gameplay scenarios. This work represents an interesting application of vision-based neuroevolution, highlighting its potential use in a wide range of challenging game environments lacking direct API support or well-defined state representations. 

**Abstract (ZH)**: 本文探讨了利用增强拓扑神经进化（NEAT）自动化暗魂（Dark Souls）游戏玩法的应用，暗魂是一款以复杂的战斗机制、动态环境和高维视觉输入为特征的著名挑战性动作角色扮演游戏。与传统的强化学习或游戏玩法规则不同，我们的方法直接从原始像素数据进化神经网络，从而避免了需要显式的游戏状态信息。为实现这一方法，我们引入了暗魂API（DSAPI）这一新型Python框架，利用实时计算机视觉技术提取关键游戏指标，包括玩家和敌人的生命状态。借助NEAT，代理能够进化出有效的战斗策略来击败游戏初始 boss  asylum demon，而无需预定义的行为或特定领域的启发式方法。实验结果表明，进化出的代理能够达到高达35%的成功率，表明神经进化在解决复杂且视觉上复杂的游戏玩法场景方面的可行性。这项工作展示了基于视觉的神经进化的有趣应用，突显了其在缺乏直接API支持或明确状态表示的高度挑战性游戏环境中的潜在用途。 

---
# Optimizing UAV Trajectories via a Simplified Close Enough TSP Approach 

**Title (ZH)**: 基于简化足够接近TSP方法的无人机轨迹优化 

**Authors**: Hiba Bederina  

**Link**: [PDF](https://arxiv.org/pdf/2507.03775)  

**Abstract**: This article explores an approach to addressing the Close Enough Traveling Salesman Problem (CETSP). The objective is to streamline the mathematical formulation by introducing reformulations that approximate the Euclidean distances and simplify the objective function. Additionally, the use of convex sets in the constraint design offers computational benefits. The proposed methodology is empirically validated on real-world CETSP instances, with the aid of computational strategies such as a fragmented CPLEX-based approach. Results demonstrate its effectiveness in managing computational resources without compromising solution quality. Furthermore, the article analyzes the behavior of the proposed mathematical formulations, providing comprehensive insights into their performance. 

**Abstract (ZH)**: 本文探索解决近似旅行商问题（CETSP）的方法。目标是通过引入逼近欧几里得距离和简化目标函数的重新表述来优化数学公式。此外，约束设计中使用凸集提供了计算上的优势。所提出的建模方法在实际CETSP实例上进行了经验验证，借助基于CPLEX的分块计算策略等计算策略。结果表明，该方法在不牺牲解决方案质量的前提下有效地管理计算资源。此外，本文分析了所提出数学公式的性能行为，提供了对其性能的全面见解。 

---
# Agent-Based Detection and Resolution of Incompleteness and Ambiguity in Interactions with Large Language Models 

**Title (ZH)**: 基于代理的 incompleteness 和 ambiguity 检测与解决方法：与大规模语言模型的交互中存在问题的处理 

**Authors**: Riya Naik, Ashwin Srinivasan, Swati Agarwal, Estrid He  

**Link**: [PDF](https://arxiv.org/pdf/2507.03726)  

**Abstract**: Many of us now treat LLMs as modern-day oracles asking it almost any kind of question. However, consulting an LLM does not have to be a single turn activity. But long multi-turn interactions can get tedious if it is simply to clarify contextual information that can be arrived at through reasoning. In this paper, we examine the use of agent-based architecture to bolster LLM-based Question-Answering systems with additional reasoning capabilities. We examine the automatic resolution of potential incompleteness or ambiguities in questions by transducers implemented using LLM-based agents. We focus on several benchmark datasets that are known to contain questions with these deficiencies to varying degrees. We equip different LLMs (GPT-3.5-Turbo and Llama-4-Scout) with agents that act as specialists in detecting and resolving deficiencies of incompleteness and ambiguity. The agents are implemented as zero-shot ReAct agents. Rather than producing an answer in a single step, the model now decides between 3 actions a) classify b) resolve c) answer. Action a) decides if the question is incomplete, ambiguous, or normal. Action b) determines if any deficiencies identified can be resolved. Action c) answers the resolved form of the question. We compare the use of LLMs with and without the use of agents with these components. Our results show benefits of agents with transducer 1) A shortening of the length of interactions with human 2) An improvement in the answer quality and 3) Explainable resolution of deficiencies in the question. On the negative side we find while it may result in additional LLM invocations and in some cases, increased latency. But on tested datasets, the benefits outweigh the costs except when questions already have sufficient context. Suggesting the agent-based approach could be a useful mechanism to harness the power of LLMs to develop more robust QA systems. 

**Abstract (ZH)**: 基于代理的架构增强基于LLM的问答系统以提供额外的推理能力：使用转换器自动解决问题中的不完整性或模糊性 

---
# Roadmap for using large language models (LLMs) to accelerate cross-disciplinary research with an example from computational biology 

**Title (ZH)**: 大型语言模型（LLMs）在加速跨学科研究中的应用 roadmap：以计算生物学为例 

**Authors**: Ruian Ke, Ruy M. Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2507.03722)  

**Abstract**: Large language models (LLMs) are powerful artificial intelligence (AI) tools transforming how research is conducted. However, their use in research has been met with skepticism, due to concerns about hallucinations, biases and potential harms to research. These emphasize the importance of clearly understanding the strengths and weaknesses of LLMs to ensure their effective and responsible use. Here, we present a roadmap for integrating LLMs into cross-disciplinary research, where effective communication, knowledge transfer and collaboration across diverse fields are essential but often challenging. We examine the capabilities and limitations of LLMs and provide a detailed computational biology case study (on modeling HIV rebound dynamics) demonstrating how iterative interactions with an LLM (ChatGPT) can facilitate interdisciplinary collaboration and research. We argue that LLMs are best used as augmentative tools within a human-in-the-loop framework. Looking forward, we envisage that the responsible use of LLMs will enhance innovative cross-disciplinary research and substantially accelerate scientific discoveries. 

**Abstract (ZH)**: 大型语言模型（LLMs）是强大的人工智能工具，正在改变研究方式。然而，它们在研究中的应用因其幻觉、偏见以及对研究潜在危害的担忧而受到质疑。这些强调了清晰理解LLMs优点和缺点的重要性，以确保其有效和负责任地使用。在这里，我们提出了一条将LLMs整合到跨学科研究中的路线图，其中有效的沟通、知识转移和跨领域合作是必要的，但往往具有挑战性。我们探讨了LLMs的能力和局限性，并通过一个详细的计算生物学案例研究（建模HIV反弹动力学）展示了如何通过与LLM（ChatGPT）的迭代互动促进跨学科合作和研究。我们认为，LLMs最好作为在人为循环框架内的辅助工具使用。展望未来，我们预见负责地使用LLMs将促进创新的跨学科研究并显著加速科学发现。 

---
# Towards Unified Neurosymbolic Reasoning on Knowledge Graphs 

**Title (ZH)**: 知识图谱上的统一神经符号推理 

**Authors**: Qika Lin, Fangzhi Xu, Hao Lu, Kai He, Rui Mao, Jun Liu, Erik Cambria, Mengling Feng  

**Link**: [PDF](https://arxiv.org/pdf/2507.03697)  

**Abstract**: Knowledge Graph (KG) reasoning has received significant attention in the fields of artificial intelligence and knowledge engineering, owing to its ability to autonomously deduce new knowledge and consequently enhance the availability and precision of downstream applications. However, current methods predominantly concentrate on a single form of neural or symbolic reasoning, failing to effectively integrate the inherent strengths of both approaches. Furthermore, the current prevalent methods primarily focus on addressing a single reasoning scenario, presenting limitations in meeting the diverse demands of real-world reasoning tasks. Unifying the neural and symbolic methods, as well as diverse reasoning scenarios in one model is challenging as there is a natural representation gap between symbolic rules and neural networks, and diverse scenarios exhibit distinct knowledge structures and specific reasoning objectives. To address these issues, we propose a unified neurosymbolic reasoning framework, namely Tunsr, for KG reasoning. Tunsr first introduces a consistent structure of reasoning graph that starts from the query entity and constantly expands subsequent nodes by iteratively searching posterior neighbors. Based on it, a forward logic message-passing mechanism is proposed to update both the propositional representations and attentions, as well as first-order logic (FOL) representations and attentions of each node. In this way, Tunsr conducts the transformation of merging multiple rules by merging possible relations at each step. Finally, the FARI algorithm is proposed to induce FOL rules by constantly performing attention calculations over the reasoning graph. Extensive experimental results on 19 datasets of four reasoning scenarios (transductive, inductive, interpolation, and extrapolation) demonstrate the effectiveness of Tunsr. 

**Abstract (ZH)**: 统一神经符号推理框架 Tunsr 用于知识图谱推理 

---
# Towards Machine Theory of Mind with Large Language Model-Augmented Inverse Planning 

**Title (ZH)**: 基于大规模语言模型增强逆规划的机器心灵理论探索 

**Authors**: Rebekah A. Gelpí, Eric Xue, William A. Cunningham  

**Link**: [PDF](https://arxiv.org/pdf/2507.03682)  

**Abstract**: We propose a hybrid approach to machine Theory of Mind (ToM) that uses large language models (LLMs) as a mechanism for generating hypotheses and likelihood functions with a Bayesian inverse planning model that computes posterior probabilities for an agent's likely mental states given its actions. Bayesian inverse planning models can accurately predict human reasoning on a variety of ToM tasks, but these models are constrained in their ability to scale these predictions to scenarios with a large number of possible hypotheses and actions. Conversely, LLM-based approaches have recently demonstrated promise in solving ToM benchmarks, but can exhibit brittleness and failures on reasoning tasks even when they pass otherwise structurally identical versions. By combining these two methods, this approach leverages the strengths of each component, closely matching optimal results on a task inspired by prior inverse planning models and improving performance relative to models that utilize LLMs alone or with chain-of-thought prompting, even with smaller LLMs that typically perform poorly on ToM tasks. We also exhibit the model's potential to predict mental states on open-ended tasks, offering a promising direction for future development of ToM models and the creation of socially intelligent generative agents. 

**Abstract (ZH)**: 我们提出了一种混合方法来研究机器心智理论（ToM），该方法利用大型语言模型（LLMs）生成假设并结合贝叶斯逆规划模型计算给定代理行为时其可能心理状态的后验概率。尽管贝叶斯逆规划模型在多种ToM任务中能准确预测人类推理，但这些模型在处理大量假设和行为的可能性场景时存在 scalability 限制。相反，基于LLM的方法最近在解决ToM基准测试方面显示出前景，但在某些推理任务中即使通过类似结构的测试也会表现出脆弱性和失败。通过结合这两种方法，本文的方法充分发挥了各个组件的优点，在受先前逆规划模型启发的任务中逼近最优结果，并且与仅使用LLM或使用链式思考提示的模型相比，即使使用通常在ToM任务中表现较差的小型LLM，也能提高性能。此外，该模型在开放任务中预测心理状态的潜力也显示出未来发展心智理论模型和创造社会智能生成代理的有前途的方向。 

---
# Large Language Models for Combinatorial Optimization: A Systematic Review 

**Title (ZH)**: 大型语言模型在组合优化中的应用：系统综述 

**Authors**: Francesca Da Ros, Michael Soprano, Luca Di Gaspero, Kevin Roitero  

**Link**: [PDF](https://arxiv.org/pdf/2507.03637)  

**Abstract**: This systematic review explores the application of Large Language Models (LLMs) in Combinatorial Optimization (CO). We report our findings using the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) guidelines. We conduct a literature search via Scopus and Google Scholar, examining over 2,000 publications. We assess publications against four inclusion and four exclusion criteria related to their language, research focus, publication year, and type. Eventually, we select 103 studies. We classify these studies into semantic categories and topics to provide a comprehensive overview of the field, including the tasks performed by LLMs, the architectures of LLMs, the existing datasets specifically designed for evaluating LLMs in CO, and the field of application. Finally, we identify future directions for leveraging LLMs in this field. 

**Abstract (ZH)**: 这篇系统审查探讨了大型语言模型（LLMs）在组合优化（CO）中的应用。我们根据系统评价和元分析 Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) 指南报告了研究发现。我们通过 Scopus 和 Google Scholar 进行文献搜索，检视了超过 2000 篇出版物。我们根据语言、研究重点、出版年份和类型四个纳入标准和四个排除标准评估这些出版物，最终选择了 103 篇研究。我们将这些研究按语义类别和主题分类，提供了一个涵盖任务、LLM 架构、专门为评估 LLMs 在 CO 中设计的数据集以及应用领域的场合同仁的全面概述。最后，我们指出了利用 LLMs 在这一领域中的未来方向。 

---
# EvoAgentX: An Automated Framework for Evolving Agentic Workflows 

**Title (ZH)**: EvoAgentX: 一种自动演化代理工作流框架 

**Authors**: Yingxu Wang, Siwei Liu, Jinyuan Fang, Zaiqiao Meng  

**Link**: [PDF](https://arxiv.org/pdf/2507.03616)  

**Abstract**: Multi-agent systems (MAS) have emerged as a powerful paradigm for orchestrating large language models (LLMs) and specialized tools to collaboratively address complex tasks. However, existing MAS frameworks often require manual workflow configuration and lack native support for dynamic evolution and performance optimization. In addition, many MAS optimization algorithms are not integrated into a unified framework. In this paper, we present EvoAgentX, an open-source platform that automates the generation, execution, and evolutionary optimization of multi-agent workflows. EvoAgentX employs a modular architecture consisting of five core layers: the basic components, agent, workflow, evolving, and evaluation layers. Specifically, within the evolving layer, EvoAgentX integrates three MAS optimization algorithms, TextGrad, AFlow, and MIPRO, to iteratively refine agent prompts, tool configurations, and workflow topologies. We evaluate EvoAgentX on HotPotQA, MBPP, and MATH for multi-hop reasoning, code generation, and mathematical problem solving, respectively, and further assess it on real-world tasks using GAIA. Experimental results show that EvoAgentX consistently achieves significant performance improvements, including a 7.44% increase in HotPotQA F1, a 10.00% improvement in MBPP pass@1, a 10.00% gain in MATH solve accuracy, and an overall accuracy improvement of up to 20.00% on GAIA. The source code is available at: this https URL 

**Abstract (ZH)**: 多智能体系统（MAS）已成为协调大型语言模型（LLMs）和专门工具以合作解决复杂任务的强大范式。然而，现有的MAS框架通常需要手动工作流配置，并缺乏对动态演进和性能优化的原生支持。此外，许多MAS优化算法并未集成到统一框架中。在本文中，我们提出了EvoAgentX，这是一个开源平台，用于自动化多智能体工作流的生成、执行和进化优化。EvoAgentX采用模块化架构，包括五个核心层：基本组件层、智能体层、工作流层、进化层和评估层。具体来说，在进化层中，EvoAgentX集成了三种MAS优化算法——TextGrad、AFlow和MIPRO，以迭代优化智能体提示、工具配置和工作流拓扑结构。我们分别在HotPotQA、MBPP和MATH上对EvoAgentX进行了评估，用于多跳推理、代码生成和数学问题求解，并进一步使用GAIA进行了实际任务评估。实验结果表明，EvoAgentX在HotPotQA F1、MBPP pass@1、MATH求解准确率和GAIA中的整体准确率分别取得了7.44%、10.00%、10.00%和最高20.00%的显著性能提升。源代码可在以下链接获取：this https URL 

---
# Benchmarking Vector, Graph and Hybrid Retrieval Augmented Generation (RAG) Pipelines for Open Radio Access Networks (ORAN) 

**Title (ZH)**: 面向开放射频接入网络（ORAN）的向量、图和混合检索增强生成（RAG）管道基准测试 

**Authors**: Sarat Ahmad, Zeinab Nezami, Maryam Hafeez, Syed Ali Raza Zaidi  

**Link**: [PDF](https://arxiv.org/pdf/2507.03608)  

**Abstract**: Generative AI (GenAI) is expected to play a pivotal role in enabling autonomous optimization in future wireless networks. Within the ORAN architecture, Large Language Models (LLMs) can be specialized to generate xApps and rApps by leveraging specifications and API definitions from the RAN Intelligent Controller (RIC) platform. However, fine-tuning base LLMs for telecom-specific tasks remains expensive and resource-intensive. Retrieval-Augmented Generation (RAG) offers a practical alternative through in-context learning, enabling domain adaptation without full retraining. While traditional RAG systems rely on vector-based retrieval, emerging variants such as GraphRAG and Hybrid GraphRAG incorporate knowledge graphs or dual retrieval strategies to support multi-hop reasoning and improve factual grounding. Despite their promise, these methods lack systematic, metric-driven evaluations, particularly in high-stakes domains such as ORAN. In this study, we conduct a comparative evaluation of Vector RAG, GraphRAG, and Hybrid GraphRAG using ORAN specifications. We assess performance across varying question complexities using established generation metrics: faithfulness, answer relevance, context relevance, and factual correctness. Results show that both GraphRAG and Hybrid GraphRAG outperform traditional RAG. Hybrid GraphRAG improves factual correctness by 8%, while GraphRAG improves context relevance by 7%. 

**Abstract (ZH)**: 生成式AI（GenAI）预计将在未来无线网络中发挥关键作用，实现自主优化。在开放式无线接入网（ORAN）架构中，大规模语言模型（LLMs）可以通过利用RAN智能控制器（RIC）平台的规范和API定义来专门生成xApps和rApps。然而，为电信特定任务微调基础LLMs仍然コスト高且资源密集。检索增强生成（RAG）通过情境学习提供了实际的替代方案，能够实现领域适应而无需完全重新训练。尽管传统的RAG系统依赖向量检索，但新兴的GraphRAG和Hybrid GraphRAG变体结合了知识图或双检索策略，支持多跳推理并提高事实相关性。尽管这些方法有其潜力，但在如ORAN这样的高风险领域，它们缺乏系统的、基于度量的评估。在本研究中，我们使用ORAN规范对Vector RAG、GraphRAG和Hybrid GraphRAG进行了比较评估。我们使用已建立的生成度量标准（忠实度、答案相关性、上下文相关性和事实正确性）评估其在不同问题复杂度下的性能。结果表明，GraphRAG和Hybrid GraphRAG均优于传统RAG。Hybrid GraphRAG将事实正确性提高了8%，而GraphRAG将上下文相关性提高了7%。 

---
# A Universal Approach to Feature Representation in Dynamic Task Assignment Problems 

**Title (ZH)**: 动态任务分配问题中特征表示的通用方法 

**Authors**: Riccardo Lo Bianco, Remco Dijkman, Wim Nuijten, Willem van Jaarsveld  

**Link**: [PDF](https://arxiv.org/pdf/2507.03579)  

**Abstract**: Dynamic task assignment concerns the optimal assignment of resources to tasks in a business process. Recently, Deep Reinforcement Learning (DRL) has been proposed as the state of the art for solving assignment problems. DRL methods usually employ a neural network (NN) as an approximator for the policy function, which ingests the state of the process and outputs a valuation of the possible assignments. However, representing the state and the possible assignments so that they can serve as inputs and outputs for a policy NN remains an open challenge, especially when tasks or resources have features with an infinite number of possible values. To solve this problem, this paper proposes a method for representing and solving assignment problems with infinite state and action spaces. In doing so, it provides three contributions: (I) A graph-based feature representation of assignment problems, which we call assignment graph; (II) A mapping from marked Colored Petri Nets to assignment graphs; (III) An adaptation of the Proximal Policy Optimization algorithm that can learn to solve assignment problems represented through assignment graphs. To evaluate the proposed representation method, we model three archetypal assignment problems ranging from finite to infinite state and action space dimensionalities. The experiments show that the method is suitable for representing and learning close-to-optimal task assignment policies regardless of the state and action space dimensionalities. 

**Abstract (ZH)**: 动态任务分配涉及在业务流程中将资源最优分配给任务的问题。最近，深度强化学习（DRL）被认为是最先进的方法，用于解决分配问题。DRL方法通常使用神经网络（NN）作为策略函数的近似器，该近似器输入过程状态并输出可能分配的价值评估。然而，如何表示状态和可能的分配，以便它们可以作为策略NN的输入和输出仍然是一项开放的挑战，特别是在任务或资源具有无限多个可能值特征的情况下。为了解决这一问题，本文提出了一种表示和解决状态和动作空间无限的分配问题的方法。在此过程中，本文提供了三项贡献：（I）一种基于图的分配问题特征表示，称为分配图；（II）一种从标记彩色Petri网到分配图的映射；（III）一种适应性策略优化算法的改编，该算法可以从通过分配图表示的分配问题中学习。为了评估提出的表现形式方法，我们建立了三个代表性的分配问题模型，涵盖了从有限到无限状态和动作空间维度的情况。实验表明，该方法适用于表示和学习近似最优的任务分配策略，无论状态和动作空间的维度如何。 

---
# Limits of Safe AI Deployment: Differentiating Oversight and Control 

**Title (ZH)**: 安全人工智能部署的界限：区分监督与控制 

**Authors**: David Manheim, Aidan Homewood  

**Link**: [PDF](https://arxiv.org/pdf/2507.03525)  

**Abstract**: Oversight and control (collectively, supervision) are often invoked as key levers for ensuring that AI systems are accountable, reliable, and able to fulfill governance and management requirements. However, the concepts are frequently conflated or insufficiently distinguished in academic and policy discourse, undermining efforts to design or evaluate systems that should remain under meaningful human supervision.
This paper undertakes a targeted critical review of literature on supervision outside of AI, along with a brief summary of past work on the topic related to AI. We then differentiate control as being ex-ante or real-time, and operational rather than policy or governance. In contrast, oversight is either a policy and governance function, or is ex-post. We suggest that control aims to prevent failures. In contrast, oversight often focuses on detection, remediation, or incentives for future prevention; all preventative oversight strategies nonetheless necessitate control.
Building on this foundation, we make three contributions. First, we propose a theoretically-informed yet policy-grounded framework that articulates the conditions under which each mechanism is possible, where they fall short, and what is required to make them meaningful in practice. Second, we outline how supervision methods should be documented and integrated into risk management, and drawing on the Microsoft Responsible AI Maturity Model, we outline a maturity model for AI supervision. Third, we explicitly highlight some boundaries of these mechanisms, including where they apply, where they fail, and where it is clear that no existing methods suffice. This foregrounds the question of whether meaningful supervision is possible in a given deployment context, and can support regulators, auditors, and practitioners in identifying both present limitations and the need for new conceptual and technical advances. 

**Abstract (ZH)**: 监督与控制（统称为监督）常被用作确保人工智能系统问责、可靠并满足治理和管理要求的关键杠杆。然而，这些概念在学术和政策讨论中往往被混淆或区分不足，削弱了设计或评估应在有意义的人类监督下运行的系统的努力。

本文对AI之外的监督文献进行了有针对性的批判性回顾，并简要总结了与AI相关的过去研究成果。我们接着区分控制为前瞻性或实时控制，操作性而非政策或治理控制。相比之下，监督要么是政策和治理功能，要么为事后。我们认为控制旨在防止故障，而监督则更多地关注检测、补救或未来预防的激励；所有预防性的监督策略依然离不开控制。

在此基础上，我们做出了三项贡献。首先，我们提出了一种理论导向且政策基础的框架，阐述了每种机制在何种条件下可行，它们有何不足以及在实践中如何使其有意义。其次，我们概述了监督方法应如何记录并整合到风险管理中，并借鉴Microsoft负责任AI成熟度模型，提出了AI监督的成熟度模型。第三，我们明确界定了这些机制的边界，包括它们适用的范围、失效的地方以及现有的方法无法满足的情况。这突显了在特定部署环境中是否可能实现有意义的监督的问题，并可支持监管者、审计师和从业者识别当前的局限性以及需要新的概念和技术进步。 

---
# REAL: Benchmarking Abilities of Large Language Models for Housing Transactions and Services 

**Title (ZH)**: REAL: 评估大型语言模型在住房交易和服务方面的能力 

**Authors**: Kexin Zhu, Yang Han  

**Link**: [PDF](https://arxiv.org/pdf/2507.03477)  

**Abstract**: The development of large language models (LLMs) has greatly promoted the progress of chatbot in multiple fields. There is an urgent need to evaluate whether LLMs can play the role of agent in housing transactions and services as well as humans. We present Real Estate Agent Large Language Model Evaluation (REAL), the first evaluation suite designed to assess the abilities of LLMs in the field of housing transactions and services. REAL comprises 5,316 high-quality evaluation entries across 4 topics: memory, comprehension, reasoning and hallucination. All these entries are organized as 14 categories to assess whether LLMs have the knowledge and ability in housing transactions and services scenario. Additionally, the REAL is used to evaluate the performance of most advanced LLMs. The experiment results indicate that LLMs still have significant room for improvement to be applied in the real estate field. 

**Abstract (ZH)**: 大型语言模型（LLMs）的发展极大地推动了聊天机器人在多个领域的进步。迫切需要评估LLMs是否能在房地产交易和服务中扮演像人类一样的代理角色。我们提出了房地产代理大型语言模型评估（REAL），这是首个专门评估LLMs在房地产交易和服务领域的效能评估套件。REAL包含5,316个高质量的评估条目，涵盖4个主题：记忆、理解、推理和虚构。所有这些条目被组织成14个类别，以评估LLMs在房地产交易和服务场景中的知识和能力。此外，REAL被用于评估最新最先进LLMs的表现。实验结果表明，LLMs在房地产领域的应用仍有显著改进空间。 

---
# Multi-Agent Reasoning for Cardiovascular Imaging Phenotype Analysis 

**Title (ZH)**: 多agent推理在心血管影像表型分析中的应用 

**Authors**: Weitong Zhang, Mengyun Qiao, Chengqi Zang, Steven Niederer, Paul M Matthews, Wenjia Bai, Bernhard Kainz  

**Link**: [PDF](https://arxiv.org/pdf/2507.03460)  

**Abstract**: Identifying the associations between imaging phenotypes and disease risk factors and outcomes is essential for understanding disease mechanisms and improving diagnosis and prognosis models. However, traditional approaches rely on human-driven hypothesis testing and selection of association factors, often overlooking complex, non-linear dependencies among imaging phenotypes and other multi-modal data. To address this, we introduce a Multi-agent Exploratory Synergy for the Heart (MESHAgents) framework that leverages large language models as agents to dynamically elicit, surface, and decide confounders and phenotypes in association studies, using cardiovascular imaging as a proof of concept. Specifically, we orchestrate a multi-disciplinary team of AI agents -- spanning cardiology, biomechanics, statistics, and clinical research -- which spontaneously generate and converge on insights through iterative, self-organizing reasoning. The framework dynamically synthesizes statistical correlations with multi-expert consensus, providing an automated pipeline for phenome-wide association studies (PheWAS). We demonstrate the system's capabilities through a population-based study of imaging phenotypes of the heart and aorta. MESHAgents autonomously uncovered correlations between imaging phenotypes and a wide range of non-imaging factors, identifying additional confounder variables beyond standard demographic factors. Validation on diagnosis tasks reveals that MESHAgents-discovered phenotypes achieve performance comparable to expert-selected phenotypes, with mean AUC differences as small as -0.004 on disease classification tasks. Notably, the recall score improves for 6 out of 9 disease types. Our framework provides clinically relevant imaging phenotypes with transparent reasoning, offering a scalable alternative to expert-driven methods. 

**Abstract (ZH)**: 利用多智能体探索协同框架（MESHAgents）识别影像表型与疾病风险因素及结局之间的关联对于理解疾病机制和改进诊断及预后模型至关重要。传统方法依赖于人工驱动的假设测试和关联因素的选择，往往会忽略影像表型与其他多模态数据之间复杂且非线性的依赖关系。为解决这一问题，我们引入了一种基于多智能体探索协同框架（MESHAgents），利用大语言模型作为智能体动态地提出、呈现和决定混杂因素和表型，在心血管影像方面作为概念验证。具体而言，我们协调了一个跨心脏科、生物力学、统计学和临床研究的多学科AI智能体团队，通过迭代的自我组织推理自发生成并收敛于见解。该框架动态地综合了多元专家共识的统计相关性，提供了一种自动化的全表型关联研究（PheWAS）管道。我们通过基于人群的心脏和主动脉影像表型研究展示了系统的功能。MESHAgents自主发现了影像表型与非影像因素之间的多种相关性，识别出标准化的demographic因素之外的额外混杂变量。在诊断任务上的验证表明，MESHAgents发现的表型在疾病分类任务上的性能与专家选择的表型相当，平均AUC差异仅为-0.004。值得注意的是，6种疾病类型的召回率有所提高。该框架提供了具有透明推理的临床相关影像表型，提供了一种可扩展的专家驱动方法替代方案。 

---
# Lessons from a Chimp: AI "Scheming" and the Quest for Ape Language 

**Title (ZH)**: 来自 chimps 的教训：AI 的“谋略”与类人猿语言探索之旅 

**Authors**: Christopher Summerfield, Lennart Luettgau, Magda Dubois, Hannah Rose Kirk, Kobi Hackenburg, Catherine Fist, Katarina Slama, Nicola Ding, Rebecca Anselmetti, Andrew Strait, Mario Giulianelli, Cozmin Ududec  

**Link**: [PDF](https://arxiv.org/pdf/2507.03409)  

**Abstract**: We examine recent research that asks whether current AI systems may be developing a capacity for "scheming" (covertly and strategically pursuing misaligned goals). We compare current research practices in this field to those adopted in the 1970s to test whether non-human primates could master natural language. We argue that there are lessons to be learned from that historical research endeavour, which was characterised by an overattribution of human traits to other agents, an excessive reliance on anecdote and descriptive analysis, and a failure to articulate a strong theoretical framework for the research. We recommend that research into AI scheming actively seeks to avoid these pitfalls. We outline some concrete steps that can be taken for this research programme to advance in a productive and scientifically rigorous fashion. 

**Abstract (ZH)**: 我们研究了最近探讨当前AI系统是否可能发展出“算计”能力（即隐蔽且策略性地追求不一致目标）的相关研究。我们将当前该领域的研究实践与20世纪70年代测试非人灵长类动物是否能掌握自然语言能力的研究做法进行了比较，认为可以从那段历史研究中吸取教训，避免过度将人类特质归因于其他代理、过于依赖个人见证和描述性分析等问题，并未能为研究制定强有力的基础理论框架。我们建议，对于AI算计的研究应积极避免这些陷阱。我们提出了若干具体步骤，以便该研究计划能够以富有成效且科学严谨的方式推进。 

---
# Artificial intelligence in drug discovery: A comprehensive review with a case study on hyperuricemia, gout arthritis, and hyperuricemic nephropathy 

**Title (ZH)**: 人工智能在药物发现中的应用：关于高尿酸血症、痛风关节炎和高尿酸肾病的综合回顾与案例研究 

**Authors**: Junwei Su, Cheng Xin, Ao Shang, Shan Wu, Zhenzhen Xie, Ruogu Xiong, Xiaoyu Xu, Cheng Zhang, Guang Chen, Yau-Tuen Chan, Guoyi Tang, Ning Wang, Yong Xu, Yibin Feng  

**Link**: [PDF](https://arxiv.org/pdf/2507.03407)  

**Abstract**: This paper systematically reviews recent advances in artificial intelligence (AI), with a particular focus on machine learning (ML), across the entire drug discovery pipeline. Due to the inherent complexity, escalating costs, prolonged timelines, and high failure rates of traditional drug discovery methods, there is a critical need to comprehensively understand how AI/ML can be effectively integrated throughout the full process. Currently available literature reviews often narrowly focus on specific phases or methodologies, neglecting the dependence between key stages such as target identification, hit screening, and lead optimization. To bridge this gap, our review provides a detailed and holistic analysis of AI/ML applications across these core phases, highlighting significant methodological advances and their impacts at each stage. We further illustrate the practical impact of these techniques through an in-depth case study focused on hyperuricemia, gout arthritis, and hyperuricemic nephropathy, highlighting real-world successes in molecular target identification and therapeutic candidate discovery. Additionally, we discuss significant challenges facing AI/ML in drug discovery and outline promising future research directions. Ultimately, this review serves as an essential orientation for researchers aiming to leverage AI/ML to overcome existing bottlenecks and accelerate drug discovery. 

**Abstract (ZH)**: 本文系统回顾了人工智能（AI）在药物发现全过程中的最新进展，特别是机器学习（ML）在其中的应用。鉴于传统药物发现方法固有的复杂性、不断上升的成本、延长的时间线以及较高的失败率，全面理解AI/ML在整个过程中如何有效集成显得尤为关键。现有的综述文献往往仅限于特定阶段或方法，忽视了目标识别、苗头筛选和先导优化等关键阶段之间的相互依赖性。为填补这一空白，本综述提供了这些核心阶段中AI/ML应用的详细和全面分析，强调了每个阶段的重大方法学进步及其影响。我们还通过专注于高尿酸血症、痛风关节炎和高尿酸性肾病的深入案例研究，展示了这些技术的实际影响，突出了分子靶点识别和治疗候选药物发现的实际成功案例。此外，我们讨论了AI/ML在药物发现中面临的重要挑战，并概述了有希望的未来研究方向。最终，本综述为希望通过利用AI/ML克服现有瓶颈并加速药物发现的研究人员提供了一个重要的指导方向。 

---
# Effects of structure on reasoning in instance-level Self-Discover 

**Title (ZH)**: 结构对Instance-Level Self-Discover推理的影响 

**Authors**: Sachith Gunasekara, Yasiru Ratnayake  

**Link**: [PDF](https://arxiv.org/pdf/2507.03347)  

**Abstract**: The drive for predictable LLM reasoning in their integration with compound systems has popularized structured outputs, yet concerns remain about performance trade-offs compared to unconstrained natural language. At the same time, training on unconstrained Chain of Thought (CoT) traces has brought about a new class of strong reasoning models that nevertheless present novel compute budget and faithfulness challenges. This paper introduces iSelf-Discover, an instance-level adaptation of the Self-Discover framework, and using it compares dynamically generated structured JSON reasoning with its unstructured counterpart. Our empirical evaluation across diverse benchmarks using state-of-the-art open-source models supports a consistent advantage for unstructured reasoning. Notably, on the complex MATH benchmark, unstructured plans achieved relative performance improvements of up to 18.90\% over structured approaches. Zero-shot unstructured iSelf-Discover variants are also shown to outperform their five-shot structured counterparts, underscoring the significance of this gap, even when structured plans are dynamically generated to ensure reasoning precedes the final answer. We further demonstrate that the optimal granularity of plan generation (instance-level vs. task-level) is context-dependent. These findings invite re-evaluation of the reliance on structured formats for complex problem-solving and how compound systems should be organized. 

**Abstract (ZH)**: 可预测的大模型推理在复合系统中的应用推动了结构化输出的流行，但与不受约束的自然语言相比，仍存在性能权衡的担忧。同时，基于不受约束的链式思考轨迹的训练带来了新的强大推理模型，但也带来了新的计算预算和忠实性挑战。本文引入了iSelf-Discover，这是一种实例级的Self-Discover框架的适应，利用它比较动态生成的结构化JSON推理与其无结构对应物。我们的跨多种基准的数据实证评估支持无结构推理的一致优势。值得注意的是，在复杂的MATH基准测试中，无结构计划相对于结构化方法的相对性能提高了多达18.90%。零样本的无结构iSelf-Discover变体也优于其五样本的结构化对应物，突显了这种差距的重要性，即使结构化计划是动态生成的以确保推理先于最终答案。我们进一步证明，计划生成的最佳粒度（实例级 vs. 任务级）依赖于上下文。这些发现促使我们重新评估在复杂问题解决中对结构化格式的依赖以及复合系统应该如何组织。 

---
# Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky 

**Title (ZH)**: 面向消歧的微调使企业级工具调用LLM更真实可靠 

**Authors**: Ashutosh Hathidara, Julien Yu, Sebastian Schreiber  

**Link**: [PDF](https://arxiv.org/pdf/2507.03336)  

**Abstract**: Large language models (LLMs) are increasingly tasked with invoking enterprise APIs, yet they routinely falter when near-duplicate tools vie for the same user intent or when required arguments are left underspecified. We introduce DiaFORGE (Dialogue Framework for Organic Response Generation & Evaluation), a disambiguation-centric, three-stage pipeline that (i) synthesizes persona-driven, multi-turn dialogues in which the assistant must distinguish among highly similar tools, (ii) performs supervised fine-tuning of open-source models with reasoning traces across 3B - 70B parameters, and (iii) evaluates real-world readiness via a dynamic suite that redeploys each model in a live agentic loop and reports end-to-end goal completion alongside conventional static metrics. On our dynamic benchmark DiaBENCH, models trained with DiaFORGE raise tool-invocation success by 27 pp over GPT-4o and by 49 pp over Claude-3.5-Sonnet, both under optimized prompting. To spur further research, we release an open corpus of 5000 production-grade enterprise API specifications paired with rigorously validated, disambiguation-focused dialogues, offering a practical blueprint for building reliable, enterprise-ready tool-calling agents. 

**Abstract (ZH)**: Large Language Models for Disambiguation-Centric Invocation of Enterprise APIs: DiaFORGE Framework and Evaluation 

---
# Exploring Object Status Recognition for Recipe Progress Tracking in Non-Visual Cooking 

**Title (ZH)**: 探索物体状态识别在非视觉烹饪过程跟踪中的应用 

**Authors**: Franklin Mingzhe Li, Kaitlyn Ng, Bin Zhu, Patrick Carrington  

**Link**: [PDF](https://arxiv.org/pdf/2507.03330)  

**Abstract**: Cooking plays a vital role in everyday independence and well-being, yet remains challenging for people with vision impairments due to limited support for tracking progress and receiving contextual feedback. Object status - the condition or transformation of ingredients and tools - offers a promising but underexplored foundation for context-aware cooking support. In this paper, we present OSCAR (Object Status Context Awareness for Recipes), a technical pipeline that explores the use of object status recognition to enable recipe progress tracking in non-visual cooking. OSCAR integrates recipe parsing, object status extraction, visual alignment with cooking steps, and time-causal modeling to support real-time step tracking. We evaluate OSCAR on 173 instructional videos and a real-world dataset of 12 non-visual cooking sessions recorded by BLV individuals in their homes. Our results show that object status consistently improves step prediction accuracy across vision-language models, and reveal key factors that impact performance in real-world conditions, such as implicit tasks, camera placement, and lighting. We contribute the pipeline of context-aware recipe progress tracking, an annotated real-world non-visual cooking dataset, and design insights to guide future context-aware assistive cooking systems. 

**Abstract (ZH)**: 烹饪在日常独立生活和福祉中扮演着重要角色，但对于视力受损人群而言，由于缺乏跟踪进度和获取上下文反馈的支持，烹饪仍具挑战性。对象状态——食材和工具的状态或变化——为基于上下文的烹饪支持提供了有前景但尚未充分开发的基础。在本文中，我们推出了OSCAR（Object Status Context Awareness for Recipes）技术流程，探讨利用对象状态识别来实现非视觉烹饪中的食谱进度追踪。OSCAR将食谱解析、对象状态提取、烹饪步骤的视觉对齐以及时间因果建模结合起来，以支持实时步骤追踪。我们在173个教学视频和由视力受损个体在家中录制的12个非视觉烹饪会话的真实世界数据集上评估了OSCAR。我们的结果显示，对象状态在视觉语言模型中一致地提高了步骤预测的准确性，并揭示了影响实际条件下性能的关键因素，如隐含任务、摄像头位置和照明。我们贡献了基于上下文的食谱进度追踪流程、标注的真实世界非视觉烹饪数据集以及设计见解，以指导未来基于上下文的辅助烹饪系统的设计。 

---
# NDAI-NeuroMAP: A Neuroscience-Specific Embedding Model for Domain-Specific Retrieval 

**Title (ZH)**: NDAI-NeuroMAP：一种神经科学专用的嵌入模型用于领域特定检索 

**Authors**: Devendra Patel, Aaditya Jain, Jayant Verma, Divyansh Rajput, Sunil Mahala, Ketki Suresh Khapare, Jayateja Kalla  

**Link**: [PDF](https://arxiv.org/pdf/2507.03329)  

**Abstract**: We present NDAI-NeuroMAP, the first neuroscience-domain-specific dense vector embedding model engineered for high-precision information retrieval tasks. Our methodology encompasses the curation of an extensive domain-specific training corpus comprising 500,000 carefully constructed triplets (query-positive-negative configurations), augmented with 250,000 neuroscience-specific definitional entries and 250,000 structured knowledge-graph triplets derived from authoritative neurological ontologies. We employ a sophisticated fine-tuning approach utilizing the FremyCompany/BioLORD-2023 foundation model, implementing a multi-objective optimization framework combining contrastive learning with triplet-based metric learning paradigms. Comprehensive evaluation on a held-out test dataset comprising approximately 24,000 neuroscience-specific queries demonstrates substantial performance improvements over state-of-the-art general-purpose and biomedical embedding models. These empirical findings underscore the critical importance of domain-specific embedding architectures for neuroscience-oriented RAG systems and related clinical natural language processing applications. 

**Abstract (ZH)**: NDAI-NeuroMAP：首个工程化用于高精度信息检索任务的神经科学领域特定密集向量嵌入模型 

---
# LTLCrit: A Temporal Logic-based LLM Critic for Safe and Efficient Embodied Agents 

**Title (ZH)**: 基于时序逻辑的LLM批评家：用于安全高效体现式代理的LTLCrit 

**Authors**: Anand Gokhale, Vaibhav Srivastava, Francesco Bullo  

**Link**: [PDF](https://arxiv.org/pdf/2507.03293)  

**Abstract**: Large language models (LLMs) have demonstrated promise in reasoning tasks and general decision-making in static environments. In long-term planning tasks, however, errors tend to accumulate, often leading to unsafe or inefficient behavior, limiting their use in general-purpose settings. We propose a modular actor-critic architecture in which an LLM actor is guided by LTLCrit, a trajectory-level LLM critic that communicates via linear temporal logic (LTL). Our setup combines the reasoning strengths of language models with the guarantees of formal logic. The actor selects high-level actions from natural language observations, while the critic analyzes full trajectories and proposes new LTL constraints that shield the actor from future unsafe or inefficient behavior. The architecture supports both fixed, hand-specified safety constraints and adaptive, learned soft constraints that promote long-term efficiency. Our architecture is model-agnostic: any LLM-based planner can serve as the actor, and LTLCrit serves as a logic-generating wrapper. We formalize planning as graph traversal under symbolic constraints, allowing LTLCrit to analyze failed or suboptimal trajectories and generate new temporal logic rules that improve future behavior. We evaluate our system on the Minecraft diamond-mining benchmark, achieving 100% completion rates and improving efficiency compared to baseline LLM planners. Our results suggest that enabling LLMs to supervise each other through logic is a powerful and flexible paradigm for safe, generalizable decision making. 

**Abstract (ZH)**: 大型语言模型在长期规划任务中的模块化actor-critic架构：通过线性时序逻辑实现安全与效率 

---
# Memory Mosaics at scale 

**Title (ZH)**: 大规模内存拼图 

**Authors**: Jianyu Zhang, Léon Bottou  

**Link**: [PDF](https://arxiv.org/pdf/2507.03285)  

**Abstract**: Memory Mosaics [Zhang et al., 2025], networks of associative memories, have demonstrated appealing compositional and in-context learning capabilities on medium-scale networks (GPT-2 scale) and synthetic small datasets. This work shows that these favorable properties remain when we scale memory mosaics to large language model sizes (llama-8B scale) and real-world datasets.
To this end, we scale memory mosaics to 10B size, we train them on one trillion tokens, we introduce a couple architectural modifications ("Memory Mosaics v2"), we assess their capabilities across three evaluation dimensions: training-knowledge storage, new-knowledge storage, and in-context learning.
Throughout the evaluation, memory mosaics v2 match transformers on the learning of training knowledge (first dimension) and significantly outperforms transformers on carrying out new tasks at inference time (second and third dimensions). These improvements cannot be easily replicated by simply increasing the training data for transformers. A memory mosaics v2 trained on one trillion tokens still perform better on these tasks than a transformer trained on eight trillion tokens. 

**Abstract (ZH)**: Memory Mosaics [张等, 2025]，关联记忆网络，在中型网络（GPT-2规模）和合成小型数据集上展示了吸引人的组合性和上下文学习能力。本研究展示了当我们将记忆拼图扩展到大型语言模型规模（llama-8B规模）和真实世界数据集时，这些有利特性依然存在。

为此，我们将记忆拼图扩展到10B规模，使用一万亿个令牌对其进行训练，引入了几种架构修改（“Memory Mosaics v2”），并在三个评估维度上对其能力进行了评估：训练知识存储、新知识存储和上下文学习。

在整个评估过程中，Memory Mosaics v2 在学习训练知识（第一个维度）上与Transformer持平，并在推断时执行新任务（第二个和第三个维度）上显著优于Transformer。这些改进无法通过简单增加Transformer的训练数据来轻易复制。即使使用一万亿个令牌训练的Memory Mosaics v2 在这些任务上也优于使用八万亿个令牌训练的Transformer。 

---
# GDGB: A Benchmark for Generative Dynamic Text-Attributed Graph Learning 

**Title (ZH)**: GDGB: 生成动态文本图学习的基准数据集 

**Authors**: Jie Peng, Jiarui Ji, Runlin Lei, Zhewei Wei, Yongchao Liu, Chuntao Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.03267)  

**Abstract**: Dynamic Text-Attributed Graphs (DyTAGs), which intricately integrate structural, temporal, and textual attributes, are crucial for modeling complex real-world systems. However, most of the existing DyTAG datasets exhibit poor textual quality, which severely limits their utility for DyTAG generation tasks requiring semantically rich inputs. Additionally, prior work mainly focuses on discriminative tasks on DyTAGs, resulting in a lack of standardized task formulations and evaluation protocols tailored for DyTAG generation. To address these critical issues, we propose Generative DyTAG Benchmark (GDGB), which comprises eight meticulously curated DyTAG datasets with high-quality textual features for both nodes and edges, overcoming limitations of prior datasets. Building on GDGB, we define two novel DyTAG generation tasks: Transductive Dynamic Graph Generation (TDGG) and Inductive Dynamic Graph Generation (IDGG). TDGG transductively generates a target DyTAG based on the given source and destination node sets, while the more challenging IDGG introduces new node generation to inductively model the dynamic expansion of real-world graph data. To enable holistic evaluation, we design multifaceted metrics that assess the structural, temporal, and textual quality of the generated DyTAGs. We further propose GAG-General, an LLM-based multi-agent generative framework tailored for reproducible and robust benchmarking of DyTAG generation. Experimental results demonstrate that GDGB enables rigorous evaluation of TDGG and IDGG, with key insights revealing the critical interplay of structural and textual features in DyTAG generation. These findings establish GDGB as a foundational resource for advancing generative DyTAG research and unlocking further practical applications in DyTAG generation. GDGB datasets, source codes, and leaderboards are available at \href{this https URL}{here}. 

**Abstract (ZH)**: 动态文本属性图（DyTAGs）：结构、时间和文本属性的精细整合对于 modeling 复杂现实系统至关重要。然而，现有的大多数 DyTAG 数据集在文本质量方面表现不佳，严重限制了其在需要丰富语义输入的 DyTAG 生成任务中的应用。此外，前人的工作主要集中在 DyTAG 的区分性任务上，导致缺乏针对 DyTAG 生成的标准任务形式和评估协议。为应对这些关键问题，我们提出生成性动态文本属性图基准（GDGB），包含八个精心策划的 DyTAG 数据集，具有高质量的节点和边的文本特征，克服了先前数据集的限制。基于 GDGB，我们定义了两个新的 DyTAG 生成任务：归纳动态图生成（IDGG）和传递动态图生成（TDGG）。TDGG 通过给定的源节点和目标节点集，生成目标 DyTAG，而更具挑战性的 IDGG 引入了新的节点生成，以归纳地建模现实世界图数据的动态扩展。为了实现全面评估，我们设计了多方面的评估指标，以评估生成的 DyTAG 的结构、时间和文本质量。我们进一步提出了 GAG-General，这是一种基于大语言模型的多智能体生成框架，专门用于 DyTAG 生成基准测试的可重复性和鲁棒性。实验结果表明，GDGB 能够严谨地评估 TDGG 和 IDGG，并揭示了 DyTAG 生成中结构和文本特征的密切互动。这些发现确立了 GDGB 作为推进生成性 DyTAG 研究和解锁进一步实际应用场景的基础资源。GDGB 数据集、源代码和排行榜可在 \href{this https URL}{这里} 获取。 

---
# CodeAgents: A Token-Efficient Framework for Codified Multi-Agent Reasoning in LLMs 

**Title (ZH)**: CodeAgents：一种用于大语言模型中编码多智能体推理的高效token框架 

**Authors**: Bruce Yang, Xinfeng He, Huan Gao, Yifan Cao, Xiaofan Li, David Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2507.03254)  

**Abstract**: Effective prompt design is essential for improving the planning capabilities of large language model (LLM)-driven agents. However, existing structured prompting strategies are typically limited to single-agent, plan-only settings, and often evaluate performance solely based on task accuracy - overlooking critical factors such as token efficiency, modularity, and scalability in multi-agent environments. To address these limitations, we introduce CodeAgents, a prompting framework that codifies multi-agent reasoning and enables structured, token-efficient planning in multi-agent systems. In CodeAgents, all components of agent interaction - Task, Plan, Feedback, system roles, and external tool invocations - are codified into modular pseudocode enriched with control structures (e.g., loops, conditionals), boolean logic, and typed variables. This design transforms loosely connected agent plans into cohesive, interpretable, and verifiable multi-agent reasoning programs. We evaluate the proposed framework across three diverse benchmarks - GAIA, HotpotQA, and VirtualHome - using a range of representative LLMs. Results show consistent improvements in planning performance, with absolute gains of 3-36 percentage points over natural language prompting baselines. On VirtualHome, our method achieves a new state-of-the-art success rate of 56%. In addition, our approach reduces input and output token usage by 55-87% and 41-70%, respectively, underscoring the importance of token-aware evaluation metrics in the development of scalable multi-agent LLM systems. The code and resources are available at: this https URL 

**Abstract (ZH)**: 有效的提示设计对于提高大型语言模型（LLM）驱动代理的规划能力至关重要。然而，现有的结构化提示策略通常仅限于单代理、仅规划的设置，并且往往仅基于任务准确性来评估性能，忽视了多代理环境中关键因素，如 token 效率、模块化和可扩展性。为解决这些问题，我们引入了 CodeAgents，这是一个编码多代理推理的提示框架，并在多代理系统中实现结构化和 token 效率的规划。在 CodeAgents 中，所有代理交互组件——任务、计划、反馈、系统角色和外部工具调用——被编码为带有控制结构（如循环、条件）、布尔逻辑和类型变量的模块化伪代码。这种设计将松散连接的代理计划转化为连贯、可解释且可验证的多代理推理程序。我们在 GAIA、HotpotQA 和 VirtualHome 三个不同的基准测试中，使用代表性的 LLMs 评估了所提出的框架。结果显示在规划性能上的一致改进，绝对收益为 3-36 个百分点，超过自然语言提示基线。在 VirtualHome 上，我们的方法达到了新的最佳成功率 56%。此外，我们的方法将输入和输出 token 使用量分别减少了 55-87% 和 41-70%，突显了在开发可扩展的多代理 LLM 系统时 token 意识评估指标的重要性。相关代码和资源可在：this https URL 获取。 

---
# Efficient Knowledge Graph Construction and Retrieval from Unstructured Text for Large-Scale RAG Systems 

**Title (ZH)**: 从未结构化文本中高效构建和检索知识图谱以支持大规模RAG系统 

**Authors**: Congmin Min, Rhea Mathew, Joyce Pan, Sahil Bansal, Abbas Keshavarzi, Amar Viswanathan Kannan  

**Link**: [PDF](https://arxiv.org/pdf/2507.03226)  

**Abstract**: We propose a scalable and cost-efficient framework for deploying Graph-based Retrieval Augmented Generation (GraphRAG) in enterprise environments. While GraphRAG has shown promise for multi-hop reasoning and structured retrieval, its adoption has been limited by the high computational cost of constructing knowledge graphs using large language models (LLMs) and the latency of graph-based retrieval. To address these challenges, we introduce two core innovations: (1) a dependency-based knowledge graph construction pipeline that leverages industrial-grade NLP libraries to extract entities and relations from unstructured text completely eliminating reliance on LLMs; and (2) a lightweight graph retrieval strategy that combines hybrid query node identification with efficient one-hop traversal for high-recall, low-latency subgraph extraction. We evaluate our framework on two SAP datasets focused on legacy code migration and demonstrate strong empirical performance. Our system achieves up to 15% and 4.35% improvements over traditional RAG baselines based on LLM-as-Judge and RAGAS metrics, respectively. Moreover, our dependency-based construction approach attains 94% of the performance of LLM-generated knowledge graphs (61.87% vs. 65.83%) while significantly reducing cost and improving scalability. These results validate the feasibility of deploying GraphRAG systems in real-world, large-scale enterprise applications without incurring prohibitive resource requirements paving the way for practical, explainable, and domain-adaptable retrieval-augmented reasoning. 

**Abstract (ZH)**: 一种可扩展且成本效益高的框架：在企业环境中部署基于图的检索增强生成（GraphRAG） 

---
# SI-Agent: An Agentic Framework for Feedback-Driven Generation and Tuning of Human-Readable System Instructions for Large Language Models 

**Title (ZH)**: SI-Agent: 一个基于代理的框架，用于大型语言模型的人工可读系统指令的反馈驱动生成与调优 

**Authors**: Jeshwanth Challagundla  

**Link**: [PDF](https://arxiv.org/pdf/2507.03223)  

**Abstract**: System Instructions (SIs), or system prompts, are pivotal for guiding Large Language Models (LLMs) but manual crafting is resource-intensive and often suboptimal. Existing automated methods frequently generate non-human-readable "soft prompts," sacrificing interpretability. This paper introduces SI-Agent, a novel agentic framework designed to automatically generate and iteratively refine human-readable SIs through a feedback-driven loop. SI-Agent employs three collaborating agents: an Instructor Agent, an Instruction Follower Agent (target LLM), and a Feedback/Reward Agent evaluating task performance and optionally SI readability. The framework utilizes iterative cycles where feedback guides the Instructor's refinement strategy (e.g., LLM-based editing, evolutionary algorithms). We detail the framework's architecture, agent roles, the iterative refinement process, and contrast it with existing methods. We present experimental results validating SI-Agent's effectiveness, focusing on metrics for task performance, SI readability, and efficiency. Our findings indicate that SI-Agent generates effective, readable SIs, offering a favorable trade-off between performance and interpretability compared to baselines. Potential implications include democratizing LLM customization and enhancing model transparency. Challenges related to computational cost and feedback reliability are acknowledged. 

**Abstract (ZH)**: 系统指令（SIs）或系统提示对于引导大型语言模型（LLMs）至关重要，但手动构建资源密集且往往不尽如人意。现有的自动化方法经常生成非人类可读的“软提示”，牺牲了可解释性。本文介绍了SI-Agent，这是一种新颖的代理框架，旨在通过反馈驱动的循环自动生成和逐步优化可读性高的SIs。SI-Agent采用三个协作代理：指导代理、指令跟随代理（目标LLM）和评估任务性能并可选评估SIs可读性的反馈/奖励代理。该框架利用迭代循环，其中反馈指导指导代理的优化策略（例如，基于LLM的编辑、进化算法）。我们详细介绍了框架的架构、代理角色、迭代优化过程，并将其与现有方法进行了对比。我们展示了实验结果验证了SI-Agent的有效性，重点关注任务性能、SI可读性和效率的度量。我们的研究表明，SI-Agent生成了有效且可读的SIs，与基准相比，在性能和可解释性之间提供了有利的权衡。潜在的影响包括使LLM定制民主化并增强模型透明度。计算成本和反馈可靠性相关挑战也得到了认可。 

---
# Discovering Algorithms with Computational Language Processing 

**Title (ZH)**: 基于计算语言处理发现算法 

**Authors**: Theo Bourdais, Abeynaya Gnanasekaran, Houman Owhadi, Tuhin Sahai  

**Link**: [PDF](https://arxiv.org/pdf/2507.03190)  

**Abstract**: Algorithms are the engine for reproducible problem-solving. We present a framework automating algorithm discovery by conceptualizing them as sequences of operations, represented as tokens. These computational tokens are chained using a grammar, enabling the formation of increasingly sophisticated procedures. Our ensemble Monte Carlo tree search (MCTS) guided by reinforcement learning (RL) explores token chaining and drives the creation of new tokens. This methodology rediscovers, improves, and generates new algorithms that substantially outperform existing methods for strongly NP-hard combinatorial optimization problems and foundational quantum computing approaches such as Grover's and Quantum Approximate Optimization Algorithm. Operating at the computational rather than code-generation level, our framework produces algorithms that can be tailored specifically to problem instances, not merely classes. 

**Abstract (ZH)**: 算法是可重复问题求解的引擎。我们提出了一种框架，通过将算法概念化为操作序列并用标记表示来自动化算法发现，这些计算标记通过语法链成，使程序越来越复杂。我们的集成蒙特卡洛树搜索（MCTS），由强化学习（RL）引导，探索标记链成并驱动新标记的生成。该方法重新发现、改进和生成了在强NP难组合优化问题和基础量子计算方法（如Grover算法和量子近似优化算法）方面显著优于现有方法的新算法。在计算层面而非代码生成层面操作，该框架产生的算法可以针对具体的问题实例进行定制，而不仅仅是类别。 

---
# LLMs are Capable of Misaligned Behavior Under Explicit Prohibition and Surveillance 

**Title (ZH)**: LLMs在明确禁止和监控下的偏离行为能力 

**Authors**: Igor Ivanov  

**Link**: [PDF](https://arxiv.org/pdf/2507.02977)  

**Abstract**: In this paper, LLMs are tasked with completing an impossible quiz, while they are in a sandbox, monitored, told about these measures and instructed not to cheat. Some frontier LLMs cheat consistently and attempt to circumvent restrictions despite everything. The results reveal a fundamental tension between goal-directed behavior and alignment in current LLMs. The code and evaluation logs are available at this http URL 

**Abstract (ZH)**: 在本文中，LLM在沙盒环境中完成一项不可能的测验，受到监控并被告知这些措施，同时被指示不要作弊。一些前沿的LLM尽管受到约束依然一致地作弊并试图规避限制。研究结果揭示了当前LLM中目标导向行为与对齐之间的基本矛盾。该代码和评估日志可在以下网址获取。 

---
# Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions 

**Title (ZH)**: 基于增量多轮交互评估LLM代理的记忆能力 

**Authors**: Yuanzhe Hu, Yu Wang, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2507.05257)  

**Abstract**: Recent benchmarks for Large Language Model (LLM) agents primarily focus on evaluating reasoning, planning, and execution capabilities, while another critical component-memory, encompassing how agents memorize, update, and retrieve long-term information-is under-evaluated due to the lack of benchmarks. We term agents with memory mechanisms as memory agents. In this paper, we identify four core competencies essential for memory agents: accurate retrieval, test-time learning, long-range understanding, and conflict resolution. Existing datasets either rely on limited context lengths or are tailored for static, long-context settings like book-based QA, which do not reflect the interactive, multi-turn nature of memory agents that incrementally accumulate information. Furthermore, no existing benchmarks cover all four competencies. Therefore, we introduce MemoryAgentBench, a new benchmark specifically designed for memory agents. Our benchmark combines reformulated existing datasets with newly constructed ones, covering the above four memory competencies, providing a systematic and challenging testbed for assessing memory quality. We evaluate a diverse set of memory agents, ranging from simple context-based and retrieval-augmented generation (RAG) systems to advanced agents with external memory modules and tool integration. Empirical results reveal that current methods fall short of mastering all four competencies, underscoring the need for further research into comprehensive memory mechanisms for LLM agents. 

**Abstract (ZH)**: Recent benchmarks for Large Language Model (LLM) agents primarily focus on evaluating reasoning, planning, and execution capabilities, while another critical component-memory, encompassing how agents memorize, update, and retrieve long-term information-is under-evaluated due to the lack of benchmarks. We term agents with memory mechanisms as memory agents. 在此论文中，我们识别出记忆代理四个核心能力：准确检索、测试时学习、长范围理解以及冲突解决。现有的数据集要么依赖于有限的上下文长度，要么针对静态的、长上下文设置（如基于书籍的问答），这些数据集都没有反映出记忆代理的交互式、多轮对话特性，记忆代理会逐步积累信息。此外，现有的基准测试并未涵盖所有四个能力。因此，我们提出了MemoryAgentBench，这是专门为记忆代理设计的新基准。我们的基准结合了重新构想的现有数据集和新构建的数据集，涵盖了上述四个记忆能力，提供了一个系统且具有挑战性的测试平台，以评估记忆质量。我们评估了一系列记忆代理，从简单的基于上下文和检索增强生成（RAG）系统到具有外部记忆模块和工具集成的高级代理。实验结果表明，现有方法在掌握所有四个能力方面存在不足，突显了对全面记忆机制进一步研究的需求。 

---
# From Marginal to Joint Predictions: Evaluating Scene-Consistent Trajectory Prediction Approaches for Automated Driving 

**Title (ZH)**: 从边缘到联合预测：评估场景一致的轨迹预测方法在自动驾驶中的性能 

**Authors**: Fabian Konstantinidis, Ariel Dallari Guerreiro, Raphael Trumpp, Moritz Sackmann, Ulrich Hofmann, Marco Caccamo, Christoph Stiller  

**Link**: [PDF](https://arxiv.org/pdf/2507.05254)  

**Abstract**: Accurate motion prediction of surrounding traffic participants is crucial for the safe and efficient operation of automated vehicles in dynamic environments. Marginal prediction models commonly forecast each agent's future trajectories independently, often leading to sub-optimal planning decisions for an automated vehicle. In contrast, joint prediction models explicitly account for the interactions between agents, yielding socially and physically consistent predictions on a scene level. However, existing approaches differ not only in their problem formulation but also in the model architectures and implementation details used, making it difficult to compare them. In this work, we systematically investigate different approaches to joint motion prediction, including post-processing of the marginal predictions, explicitly training the model for joint predictions, and framing the problem as a generative task. We evaluate each approach in terms of prediction accuracy, multi-modality, and inference efficiency, offering a comprehensive analysis of the strengths and limitations of each approach. Several prediction examples are available at this https URL. 

**Abstract (ZH)**: 周围交通参与者的准确运动预测对于动态环境中自动驾驶车辆的安全和高效运行至关重要。边际预测模型通常独立预测每个代理的未来轨迹， often导致对自动驾驶车辆的次优规划决策。相比之下，联合预测模型明确考虑了代理之间的交互，从而在场景级别提供了社会上和物理上一致的预测。然而，现有方法不仅在问题表述上有所不同，也在所使用的模型架构和实施细节上有所不同，这使得它们难以比较。本工作中，我们系统地探讨了不同的联合运动预测方法，包括对边际预测进行后处理、明确训练模型进行联合预测以及将问题建模为生成任务。我们从预测准确性、多模态性和推理效率等方面评估每个方法，提供了对每种方法优缺点的全面分析。更多信息请访问此网址：此 https URL。 

---
# Action Space Reduction Strategies for Reinforcement Learning in Autonomous Driving 

**Title (ZH)**: 自主驾驶中强化学习的动作空间缩减策略 

**Authors**: Elahe Delavari, Feeza Khan Khanzada, Jaerock Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2507.05251)  

**Abstract**: Reinforcement Learning (RL) offers a promising framework for autonomous driving by enabling agents to learn control policies through interaction with environments. However, large and high-dimensional action spaces often used to support fine-grained control can impede training efficiency and increase exploration costs. In this study, we introduce and evaluate two novel structured action space modification strategies for RL in autonomous driving: dynamic masking and relative action space reduction. These approaches are systematically compared against fixed reduction schemes and full action space baselines to assess their impact on policy learning and performance. Our framework leverages a multimodal Proximal Policy Optimization agent that processes both semantic image sequences and scalar vehicle states. The proposed dynamic and relative strategies incorporate real-time action masking based on context and state transitions, preserving action consistency while eliminating invalid or suboptimal choices. Through comprehensive experiments across diverse driving routes, we show that action space reduction significantly improves training stability and policy performance. The dynamic and relative schemes, in particular, achieve a favorable balance between learning speed, control precision, and generalization. These findings highlight the importance of context-aware action space design for scalable and reliable RL in autonomous driving tasks. 

**Abstract (ZH)**: 强化学习（RL）为自主驾驶提供了有前途的框架，通过使代理通过与环境的交互来学习控制策略。然而，用于支持精细控制的大型和高维动作空间往往会阻碍训练效率并增加探索成本。在本研究中，我们引入并评估了两种新的强化学习在自主驾驶中的结构化动作空间修改策略：动态遮蔽和相对动作空间缩减。这些方法系统地与固定缩减方案和全动作空间基线进行比较，以评估其对策略学习和性能的影响。我们的框架采用了多模态的密切策略优化代理，处理语义图像序列和标量车辆状态。所提出的动态和相对策略基于上下文和状态转换实时遮蔽动作，保持动作一致性同时消除无效或次优选择。通过在多种驾驶路线上的全面实验，我们展示了动作空间缩减显著提高了训练稳定性和策略性能。特别是，动态和相对方案在学习速度、控制精度和泛化能力之间取得了有利的平衡。这些发现强调了用于可扩展和可靠的自主驾驶任务中强化学习的上下文感知动作空间设计的重要性。 

---
# CTA: Cross-Task Alignment for Better Test Time Training 

**Title (ZH)**: CTA：跨任务对齐以提高测试时训练效果 

**Authors**: Samuel Barbeau, Pedram Fekri, David Osowiechi, Ali Bahri, Moslem YazdanpanahMasih Aminbeidokhti, Christian Desrosiers  

**Link**: [PDF](https://arxiv.org/pdf/2507.05221)  

**Abstract**: Deep learning models have demonstrated exceptional performance across a wide range of computer vision tasks. However, their performance often degrades significantly when faced with distribution shifts, such as domain or dataset changes. Test-Time Training (TTT) has emerged as an effective method to enhance model robustness by incorporating an auxiliary unsupervised task during training and leveraging it for model updates at test time. In this work, we introduce CTA (Cross-Task Alignment), a novel approach for improving TTT. Unlike existing TTT methods, CTA does not require a specialized model architecture and instead takes inspiration from the success of multi-modal contrastive learning to align a supervised encoder with a self-supervised one. This process enforces alignment between the learned representations of both models, thereby mitigating the risk of gradient interference, preserving the intrinsic robustness of self-supervised learning and enabling more semantically meaningful updates at test-time. Experimental results demonstrate substantial improvements in robustness and generalization over the state-of-the-art on several benchmark datasets. 

**Abstract (ZH)**: 深度学习模型在广泛计算机视觉任务中展现了出色的性能。然而，它们在面对分布变化，如领域或数据集变化时，性能往往会显著下降。测试时训练（TTT）作为一种有效方法，通过在训练过程中引入辅助无监督任务，并在测试时利用该任务对模型进行更新，提升了模型的鲁棒性。本文提出了一种名为CTA（Cross-Task Alignment）的新方法，以进一步改进TTT。与现有的TTT方法不同，CTA 不需要特定的模型结构，而是从多模态对比学习的成功中汲取灵感，将监督编码器与自监督编码器对齐。这一过程确保两种模型学习表示之间的对齐，从而减轻梯度干扰的风险，保留自监督学习的内在鲁棒性，并在测试时实现更具语义意义的更新。实验结果在多个基准数据集上展示了在鲁棒性和泛化能力上的显著改进。 

---
# All in One: Visual-Description-Guided Unified Point Cloud Segmentation 

**Title (ZH)**: 一气呵成：视觉描述引导的统一点云分割 

**Authors**: Zongyan Han, Mohamed El Amine Boudjoghra, Jiahua Dong, Jinhong Wang, Rao Muhammad Anwer  

**Link**: [PDF](https://arxiv.org/pdf/2507.05211)  

**Abstract**: Unified segmentation of 3D point clouds is crucial for scene understanding, but is hindered by its sparse structure, limited annotations, and the challenge of distinguishing fine-grained object classes in complex environments. Existing methods often struggle to capture rich semantic and contextual information due to limited supervision and a lack of diverse multimodal cues, leading to suboptimal differentiation of classes and instances. To address these challenges, we propose VDG-Uni3DSeg, a novel framework that integrates pre-trained vision-language models (e.g., CLIP) and large language models (LLMs) to enhance 3D segmentation. By leveraging LLM-generated textual descriptions and reference images from the internet, our method incorporates rich multimodal cues, facilitating fine-grained class and instance separation. We further design a Semantic-Visual Contrastive Loss to align point features with multimodal queries and a Spatial Enhanced Module to model scene-wide relationships efficiently. Operating within a closed-set paradigm that utilizes multimodal knowledge generated offline, VDG-Uni3DSeg achieves state-of-the-art results in semantic, instance, and panoptic segmentation, offering a scalable and practical solution for 3D understanding. Our code is available at this https URL. 

**Abstract (ZH)**: 统一分割3D点云对于场景理解至关重要，但由于其稀疏结构、有限的标注以及复杂环境中细粒度对象类别的区分挑战，这一过程受到阻碍。现有方法往往由于监督不足和缺乏多元模态提示，难以捕获丰富的语义和上下文信息，导致类别和实例区分不理想。为应对这些挑战，我们提出VDG-Uni3DSeg，这是一种新颖的框架，结合了预训练的图象-语言模型（例如CLIP）和大型语言模型（LLMs）以增强3D分割。通过利用LLM生成的文本描述和互联网上的参考图像，我们的方法整合了丰富的多元模态提示，促进细粒度类别和实例的分离。我们进一步设计了语义-视觉对比损失以对齐点特征和多元模态查询，并设计了空间增强模块以高效建模场景范围内的关系。在利用离线生成的多元模态知识的封闭集框架内，VDG-Uni3DSeg在语义、实例和泛光分割方面取得了最先进的结果，提供了一种可扩展且实用的3D理解解决方案。我们的代码可在以下链接获得。 

---
# EmbodieDreamer: Advancing Real2Sim2Real Transfer for Policy Training via Embodied World Modeling 

**Title (ZH)**: EmbodieDreamer: 通过具身世界建模促进从真实到模拟再到真实的策略训练转移 

**Authors**: Boyuan Wang, Xinpan Meng, Xiaofeng Wang, Zheng Zhu, Angen Ye, Yang Wang, Zhiqin Yang, Chaojun Ni, Guan Huang, Xingang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05198)  

**Abstract**: The rapid advancement of Embodied AI has led to an increasing demand for large-scale, high-quality real-world data. However, collecting such embodied data remains costly and inefficient. As a result, simulation environments have become a crucial surrogate for training robot policies. Yet, the significant Real2Sim2Real gap remains a critical bottleneck, particularly in terms of physical dynamics and visual appearance. To address this challenge, we propose EmbodieDreamer, a novel framework that reduces the Real2Sim2Real gap from both the physics and appearance perspectives. Specifically, we propose PhysAligner, a differentiable physics module designed to reduce the Real2Sim physical gap. It jointly optimizes robot-specific parameters such as control gains and friction coefficients to better align simulated dynamics with real-world observations. In addition, we introduce VisAligner, which incorporates a conditional video diffusion model to bridge the Sim2Real appearance gap by translating low-fidelity simulated renderings into photorealistic videos conditioned on simulation states, enabling high-fidelity visual transfer. Extensive experiments validate the effectiveness of EmbodieDreamer. The proposed PhysAligner reduces physical parameter estimation error by 3.74% compared to simulated annealing methods while improving optimization speed by 89.91\%. Moreover, training robot policies in the generated photorealistic environment leads to a 29.17% improvement in the average task success rate across real-world tasks after reinforcement learning. Code, model and data will be publicly available. 

**Abstract (ZH)**: 基于物理和视觉的Embodied AI从现实到模拟再到现实的差距缩小方法 

---
# Train-before-Test Harmonizes Language Model Rankings 

**Title (ZH)**: 训练-测试一致化语言模型排名 

**Authors**: Guanhua Zhang, Ricardo Dominguez-Olmedo, Moritz Hardt  

**Link**: [PDF](https://arxiv.org/pdf/2507.05195)  

**Abstract**: Existing language model benchmarks provide contradictory model rankings, even for benchmarks that aim to capture similar skills. This dilemma of conflicting rankings hampers model selection, clouds model comparisons, and adds confusion to a growing ecosystem of competing models. Recent work attributed ranking disagreement to the phenomenon of training on the test task: As released, different models exhibit a different level of preparation for any given test task. A candidate solution to the problem is train-before-test: Give each model the same benchmark-specific finetuning before evaluation. Our primary contribution is a broad empirical evaluation of train-before-test across 24 benchmarks and 61 models. We show that train-before-test significantly improves ranking agreement consistently across all benchmarks. Whereas rankings have little external validity to start with, they enjoy a significant degree of external validity when applying train-before-test: Model rankings transfer gracefully from one benchmark to the other. Even within the same model family, train-before-test reduces strong ranking disagreement to near-perfect agreement. In addition, train-before-test reduces the model-score matrix to essentially rank one, revealing new insights into the latent factors of benchmark performance. Our work supports the recommendation to make train-before-test a default component of LLM benchmarking. 

**Abstract (ZH)**: 现有的语言模型基准提供了矛盾的模型排名，即使是旨在捕捉相似技能的基准也不例外。这种排名冲突阻碍了模型选择，模糊了模型比较，并导致竞争模型生态系统中出现混淆。近期的工作将排名分歧归因于测试任务上的训练现象：刚发布时，不同的模型针对任何给定的测试任务都呈现出不同的准备程度。解决问题的一个候选方案是“训练后再测试”：在评估前，让每个模型进行相同的基准特定微调。我们的主要贡献是对24个基准和61个模型进行了广泛的实证评估，表明“训练后再测试”显著改善了所有基准上的排名一致性。即使排名本身一开始缺乏外部有效性，应用“训练后再测试”后，排名在不同基准之间表现出明显的外部有效性：模型排名从一个基准平滑地转移到另一个基准。即使是同一模型家族内，“训练后再测试”也将强烈排名分歧减少到几乎完美的共识。此外，“训练后再测试”将模型评分矩阵简化为几乎只有单一排名，揭示了基准性能背后潜在因素的新见解。我们的工作支持将“训练后再测试”作为大规模语言模型基准测试的默认组成部分的建议。 

---
# Infrastructuring Contestability: A Framework for Community-Defined AI Value Pluralism 

**Title (ZH)**: 社区定义的人工智能多元价值基础设施 Contestability：一种框架 

**Authors**: Andreas Mayer  

**Link**: [PDF](https://arxiv.org/pdf/2507.05187)  

**Abstract**: The proliferation of AI-driven systems presents a fundamental challenge to Human-Computer Interaction (HCI) and Computer-Supported Cooperative Work (CSCW), often diminishing user agency and failing to account for value pluralism. Current approaches to value alignment, which rely on centralized, top-down definitions, lack the mechanisms for meaningful contestability. This leaves users and communities unable to challenge or shape the values embedded in the systems that govern their digital lives, creating a crisis of legitimacy and trust. This paper introduces Community-Defined AI Value Pluralism (CDAVP), a socio-technical framework that addresses this gap. It reframes the design problem from achieving a single aligned state to infrastructuring a dynamic ecosystem for value deliberation and application. At its core, CDAVP enables diverse, self-organizing communities to define and maintain explicit value profiles - rich, machine-readable representations that can encompass not only preferences but also community-specific rights and duties. These profiles are then contextually activated by the end-user, who retains ultimate control (agency) over which values guide the AI's behavior. AI applications, in turn, are designed to transparently interpret these profiles and moderate conflicts, adhering to a set of non-negotiable, democratically-legitimated meta-rules. The designer's role shifts from crafting static interfaces to becoming an architect of participatory ecosystems. We argue that infrastructuring for pluralism is a necessary pathway toward achieving robust algorithmic accountability and genuinely contestable, human-centric AI. 

**Abstract (ZH)**: 基于社区定义的AI价值多元主义：实现 robust算法问责制与以人为本的人工智能 

---
# CREW-WILDFIRE: Benchmarking Agentic Multi-Agent Collaborations at Scale 

**Title (ZH)**: CREW-WILDFIRE：大规模评估有能动性的多智能体合作 

**Authors**: Jonathan Hyun, Nicholas R Waytowich, Boyuan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.05178)  

**Abstract**: Despite rapid progress in large language model (LLM)-based multi-agent systems, current benchmarks fall short in evaluating their scalability, robustness, and coordination capabilities in complex, dynamic, real-world tasks. Existing environments typically focus on small-scale, fully observable, or low-complexity domains, limiting their utility for developing and assessing next-generation multi-agent Agentic AI frameworks. We introduce CREW-Wildfire, an open-source benchmark designed to close this gap. Built atop the human-AI teaming CREW simulation platform, CREW-Wildfire offers procedurally generated wildfire response scenarios featuring large maps, heterogeneous agents, partial observability, stochastic dynamics, and long-horizon planning objectives. The environment supports both low-level control and high-level natural language interactions through modular Perception and Execution modules. We implement and evaluate several state-of-the-art LLM-based multi-agent Agentic AI frameworks, uncovering significant performance gaps that highlight the unsolved challenges in large-scale coordination, communication, spatial reasoning, and long-horizon planning under uncertainty. By providing more realistic complexity, scalable architecture, and behavioral evaluation metrics, CREW-Wildfire establishes a critical foundation for advancing research in scalable multi-agent Agentic intelligence. All code, environments, data, and baselines will be released to support future research in this emerging domain. 

**Abstract (ZH)**: 尽管基于大规模语言模型（LLM）的多代理系统取得了快速进展，当前的基准测试在评估其在复杂、动态的真实世界任务中的扩展性、鲁棒性和协调能力方面仍显不足。现有环境通常侧重于小规模、完全可观测或低复杂度领域，限制了其在开发和评估下一代多代理类人工智能框架方面的效用。我们引入了CREW-Wildfire，一个开源基准测试，旨在弥补这一缺口。基于人类-人工智能团队协作的CREW模拟平台，CREW-Wildfire提供了基于程序生成的wildfire响应场景，包括大规模地图、异质代理、部分可观测性、随机动力学和长期规划目标。该环境通过模块化的感知和执行模块支持低级控制和高级自然语言交互。我们实现了并评估了多个最先进的基于LLM的多代理类人工智能框架，揭示了显著的性能差距，突显了大规模协调、通信、空间推理和不确定性下的长期规划方面的未解挑战。通过提供更符合实际复杂性的架构、可扩展性以及行为评估指标，CREW-Wildfire为推进可扩展多代理类人工智能研究奠定了关键基础。所有代码、环境、数据和基线将向未来的研究开放。 

---
# OpenS2S: Advancing Open-Source End-to-End Empathetic Large Speech Language Model 

**Title (ZH)**: OpenS2S: 推动开源端到端共情大规模语音语言模型 

**Authors**: Chen Wang, Tianyu Peng, Wen Yang, Yinan Bai, Guangfu Wang, Jun Lin, Lanpeng Jia, Lingxiang Wu, Jinqiao Wang, Chengqing Zong, Jiajun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05177)  

**Abstract**: Empathetic interaction is a cornerstone of human-machine communication, due to the need for understanding speech enriched with paralinguistic cues and generating emotional and expressive responses. However, the most powerful empathetic LSLMs are increasingly closed off, leaving the crucial details about the architecture, data and development opaque to researchers. Given the critical need for transparent research into the LSLMs and empathetic behavior, we present OpenS2S, a fully open-source, transparent and end-to-end LSLM designed to enable empathetic speech interactions. Based on our empathetic speech-to-text model BLSP-Emo, OpenS2S further employs a streaming interleaved decoding architecture to achieve low-latency speech generation. To facilitate end-to-end training, OpenS2S incorporates an automated data construction pipeline that synthesizes diverse, high-quality empathetic speech dialogues at low cost. By leveraging large language models to generate empathetic content and controllable text-to-speech systems to introduce speaker and emotional variation, we construct a scalable training corpus with rich paralinguistic diversity and minimal human supervision. We release the fully open-source OpenS2S model, including the dataset, model weights, pre-training and fine-tuning codes, to empower the broader research community and accelerate innovation in empathetic speech systems. The project webpage can be accessed at this https URL 

**Abstract (ZH)**: 同理心交互是人机通信的基石，由于需要理解伴有副语言线索的声音，并生成情感表达的响应。然而，最具影响力的同理心LSLM愈加封闭，使研究人员无法获得关键的架构、数据和开发细节。鉴于对于透明研究同理心LSLM和行为的迫切需求，我们提出OpenS2S，这是一个完全开源、透明且端到端的LSLM，旨在促进同理心语音交互。基于我们的情感化的语音转文本模型BLSP-Emo，OpenS2S进一步采用流式交织解码架构以实现低延迟语音生成。为了便于端到端训练，OpenS2S整合了一个自动数据构建管道，以低成本合成丰富多样且高质量的情感化语音对话。通过利用大规模语言模型生成同理心内容，并结合可控文本转语音系统引入说话人和情感变化，我们构建了一个具有丰富副语言多样性和最少人工监督的可扩展训练语料库。我们发布了完全开源的OpenS2S模型，包括数据集、模型权重、预训练和微调代码，以赋能更广泛的科研社区并加速同理心语音系统领域的创新。项目网页可访问此链接：[this https URL] 

---
# Critiques of World Models 

**Title (ZH)**: 世界模型的批判 

**Authors**: Eric Xing, Mingkai Deng, Jinyu Hou, Zhiting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.05169)  

**Abstract**: World Model, the supposed algorithmic surrogate of the real-world environment which biological agents experience with and act upon, has been an emerging topic in recent years because of the rising needs to develop virtual agents with artificial (general) intelligence. There has been much debate on what a world model really is, how to build it, how to use it, and how to evaluate it. In this essay, starting from the imagination in the famed Sci-Fi classic Dune, and drawing inspiration from the concept of "hypothetical thinking" in psychology literature, we offer critiques of several schools of thoughts on world modeling, and argue the primary goal of a world model to be simulating all actionable possibilities of the real world for purposeful reasoning and acting. Building on the critiques, we propose a new architecture for a general-purpose world model, based on hierarchical, multi-level, and mixed continuous/discrete representations, and a generative and self-supervision learning framework, with an outlook of a Physical, Agentic, and Nested (PAN) AGI system enabled by such a model. 

**Abstract (ZH)**: 世界模型：作为一种生物代理体验和作用于真实世界环境的算法替代品，近年来由于开发具有人工（通用）智能的虚拟代理的需求增加，已经成为一个新兴话题。关于世界模型的本质、构建方法、使用方式以及评估标准，存在着诸多争论。本文从著名的科幻经典《徐古》中的想象出发，借鉴心理学文献中的“假设思维”概念，对几种世界建模学派的观点进行了批判，并提出世界模型的主要目标是在目的性推理和行动中模拟所有可行动的可能性。基于这些批判，我们提出了一种新的通用世界模型架构，基于分层、多级和混合连续/离散表示，并提出了一种生成性和自我监督学习框架，展望在这种模型支持下能够实现具身、物理和嵌套（PAN）的人工通用智能系统。 

---
# LAID: Lightweight AI-Generated Image Detection in Spatial and Spectral Domains 

**Title (ZH)**: 轻量级空间和谱域人工智能生成图像检测方法 

**Authors**: Nicholas Chivaran, Jianbing Ni  

**Link**: [PDF](https://arxiv.org/pdf/2507.05162)  

**Abstract**: The recent proliferation of photorealistic AI-generated images (AIGI) has raised urgent concerns about their potential misuse, particularly on social media platforms. Current state-of-the-art AIGI detection methods typically rely on large, deep neural architectures, creating significant computational barriers to real-time, large-scale deployment on platforms like social media. To challenge this reliance on computationally intensive models, we introduce LAID, the first framework -- to our knowledge -- that benchmarks and evaluates the detection performance and efficiency of off-the-shelf lightweight neural networks. In this framework, we comprehensively train and evaluate selected models on a representative subset of the GenImage dataset across spatial, spectral, and fusion image domains. Our results demonstrate that lightweight models can achieve competitive accuracy, even under adversarial conditions, while incurring substantially lower memory and computation costs compared to current state-of-the-art methods. This study offers valuable insight into the trade-off between efficiency and performance in AIGI detection and lays a foundation for the development of practical, scalable, and trustworthy detection systems. The source code of LAID can be found at: this https URL. 

**Abstract (ZH)**: 近期生成的高保真度人工智能图像（AIGI）在社交媒体平台上的泛滥引发了对其潜在滥用的急切担忧。当前最先进的AIGI检测方法通常依赖于大型深度神经架构，这在实时、大规模部署到如社交媒体平台时造成了显著的计算障碍。为挑战对计算密集型模型的依赖，我们引入了LAID框架，这是迄今为止首个用于评估商用轻量级神经网络检测性能和效率的基准框架。在该框架中，我们对GenImage数据集的代表性子集在空间、光谱和融合图像域进行了全面的训练和评估。研究结果表明，轻量级模型即使在对抗性条件下也能达到竞争力的准确性，同时在内存和计算成本上显著低于当前最先进的方法。本研究为AIGI检测中的效率与性能之间的权衡提供了宝贵的见解，并为开发实际可行、可扩展和可信的检测系统奠定了基础。LAID的源代码可在以下链接找到：this https URL。 

---
# AI Generated Text Detection Using Instruction Fine-tuned Large Language and Transformer-Based Models 

**Title (ZH)**: 使用指令微调大型语言模型和变压器模型生成的文本检测 

**Authors**: Chinnappa Guggilla, Budhaditya Roy, Trupti Ramdas Chavan, Abdul Rahman, Edward Bowen  

**Link**: [PDF](https://arxiv.org/pdf/2507.05157)  

**Abstract**: Large Language Models (LLMs) possess an extraordinary capability to produce text that is not only coherent and contextually relevant but also strikingly similar to human writing. They adapt to various styles and genres, producing content that is both grammatically correct and semantically meaningful. Recently, LLMs have been misused to create highly realistic phishing emails, spread fake news, generate code to automate cyber crime, and write fraudulent scientific articles. Additionally, in many real-world applications, the generated content including style and topic and the generator model are not known beforehand. The increasing prevalence and sophistication of artificial intelligence (AI)-generated texts have made their detection progressively more challenging. Various attempts have been made to distinguish machine-generated text from human-authored content using linguistic, statistical, machine learning, and ensemble-based approaches. This work focuses on two primary objectives Task-A, which involves distinguishing human-written text from machine-generated text, and Task-B, which attempts to identify the specific LLM model responsible for the generation. Both of these tasks are based on fine tuning of Generative Pre-trained Transformer (GPT_4o-mini), Large Language Model Meta AI (LLaMA) 3 8B, and Bidirectional Encoder Representations from Transformers (BERT). The fine-tuned version of GPT_4o-mini and the BERT model has achieved accuracies of 0.9547 for Task-A and 0.4698 for Task-B. 

**Abstract (ZH)**: 大型语言模型（LLMs）拥有生成连贯且上下文相关、风格和体裁上类似人类写作的文本的非凡能力。它们能够适应各种风格和体裁，产出语法正确且语义有意义的内容。最近，LLMs 被滥用以生成高度逼真的钓鱼邮件、传播假新闻、生成自动化网络犯罪的代码，以及撰写虚假的科学文章。此外，在许多实际应用中，生成的内容及其风格和主题以及生成器模型事先未知。随着人工生成文本的日益增多和复杂度提高，检测其变得越来越具有挑战性。已有多种尝试使用语言学、统计学、机器学习和集成方法来区分机器生成的文本与人工撰写的文本。本研究主要集中在两个目标上：任务-A，区分人类撰写的文本与机器生成的文本；任务-B，识别具体的生成模型。这两个任务均基于对生成预训练变换器（GPT_4o-mini）、大型语言模型Meta AI（LLaMA 3 8B）和双向编码器表示变换器（BERT）的微调。微调后的GPT_4o-mini和BERT模型在任务-A上的准确率为0.9547，在任务-B上的准确率为0.4698。 

---
# Effects of Unplanned Incoming Flights on Airport Relief Processes after a Major Natural Disaster 

**Title (ZH)**: 重大自然灾害后机场应对非计划降落航班的影响研究 

**Authors**: Luka Van de Sype, Matthieu Vert, Alexei Sharpanskykh, Seyed Sahand Mohammadi Ziabari  

**Link**: [PDF](https://arxiv.org/pdf/2507.05150)  

**Abstract**: The severity of natural disasters is increasing every year, impacting many people's lives. During the response phase of disasters, airports are important hubs where relief aid arrives and people need to be evacuated. However, the airport often forms a bottleneck in these relief operations due to the sudden need for increased capacity. Limited research has been done on the operational side of airport disaster management. Experts identify the main problems as, first, the asymmetry of information between the airport and incoming flights, and second, the lack of resources. The goal of this research is to understand the effects of incomplete knowledge of incoming flights with different resource allocation strategies on the performance of cargo handling operations at an airport after a natural disaster. An agent-based model is created, implementing realistic offloading strategies with different degrees of information uncertainty. Model calibration and verification are performed with experts in the field. The model performance is measured by the average turnaround time, which is divided into offloading time, boarding time, and cumulative waiting times. The results show that the effects of one unplanned aircraft are negligible. However, all waiting times increase with more arriving unplanned aircraft. 

**Abstract (ZH)**: 自然灾害的严重性逐年增加，影响着许多人的生活。在灾害应对阶段，机场是救济物资到达和人员疏散的重要枢纽。然而，由于容量突然增加的需求，机场往往会成为这些救济行动中的瓶颈。关于机场灾害管理的操作方面，有限的研究已经被进行。专家认为主要问题有两点：一是机场与抵达航班之间信息的不对称性，二是资源的缺乏。本研究旨在了解不同资源配置策略对机场灾害后货物处理操作性能的影响，特别是缺乏抵达航班信息的知识不完整性的影响。创建了一个基于代理的模型，实施具有不同程度信息不确定性的真实卸载策略，并与该领域的专家进行模型校准和验证。模型性能通过平均周转时间衡量，分为卸货时间、装机时间以及累计等待时间。研究结果显示，一次未计划航班的影响可以忽略不计，但随着更多未计划航班的到达，所有等待时间都会增加。 

---
# OGF: An Online Gradient Flow Method for Optimizing the Statistical Steady-State Time Averages of Unsteady Turbulent Flows 

**Title (ZH)**: OGF：优化不定湍流流统计稳态时间平均值的在线梯度流方法 

**Authors**: Tom Hickling, Jonathan F. MacArt, Justin Sirignano, Den Waidmann  

**Link**: [PDF](https://arxiv.org/pdf/2507.05149)  

**Abstract**: Turbulent flows are chaotic and unsteady, but their statistical distribution converges to a statistical steady state. Engineering quantities of interest typically take the form of time-average statistics such as $ \frac{1}{t} \int_0^t f ( u(x,\tau; \theta) ) d\tau \overset{t \rightarrow \infty}{\rightarrow} F(x; \theta)$, where $u(x,t; \theta)$ are solutions of the Navier--Stokes equations with parameters $\theta$. Optimizing over $F(x; \theta)$ has many engineering applications including geometric optimization, flow control, and closure modeling. However, this remains an open challenge, as existing computational approaches are incapable of scaling to physically representative numbers of grid points. The fundamental obstacle is the chaoticity of turbulent flows: gradients calculated with the adjoint method diverge exponentially as $t \rightarrow \infty$.
We develop a new online gradient-flow (OGF) method that is scalable to large degree-of-freedom systems and enables optimizing for the steady-state statistics of chaotic, unsteady, turbulence-resolving simulations. The method forward-propagates an online estimate for the gradient of $F(x; \theta)$ while simultaneously performing online updates of the parameters $\theta$. A key feature is the fully online nature of the algorithm to facilitate faster optimization progress and its combination with a finite-difference estimator to avoid the divergence of gradients due to chaoticity. The proposed OGF method is demonstrated for optimizations over three chaotic ordinary and partial differential equations: the Lorenz-63 equation, the Kuramoto--Sivashinsky equation, and Navier--Stokes solutions of compressible, forced, homogeneous isotropic turbulence. In each case, the OGF method successfully reduces the loss based on $F(x; \theta)$ by several orders of magnitude and accurately recovers the optimal parameters. 

**Abstract (ZH)**: 湍流流动是混沌且不稳定的，但其统计分布趋于统计稳态。感兴趣的工程量通常表现为时间平均统计量，如$\frac{1}{t} \int_0^t f ( u(x,\tau; \theta) ) d\tau \overset{t \rightarrow \infty}{\rightarrow} F(x; \theta)$，其中$u(x,t; \theta)$是具有参数$\theta$的纳维-斯托克斯方程的解。对$F(x; \theta)$进行优化具有许多工程应用，包括几何优化、流控制和闭合模型。然而，这仍然是一个待解的问题，因为现有的计算方法无法扩展到物理上代表性的网格点数量。基本障碍是湍流流动的混沌性：使用伴随方法计算的梯度随着$t \rightarrow \infty$呈指数发散。

我们开发了一种新的在线梯度流（OGF）方法，该方法可扩展到大自由度系统，并能够优化混沌、不稳定的湍流分辨率模拟的稳态统计量。该方法向前传播$F(x; \theta)$的在线梯度估计值，同时进行参数$\theta$的在线更新。一个关键特征是该算法的完全在线性质，以促进更快的优化进度，并与差分估计算法结合使用，避免因混沌性而导致梯度发散。所提出的OGF方法在三个混沌常微分方程和偏微分方程优化中得到了演示：洛伦兹-63方程、库拉모托-西瓦什金斯基方程及可压缩、强迫、各向同性的纳维-斯托克斯方程解。在每种情况下，OGF方法成功地将基于$F(x; \theta)$的损失量减少了几个数量级，并准确地恢复了最优参数。 

---
# Interpretable Mnemonic Generation for Kanji Learning via Expectation-Maximization 

**Title (ZH)**: 基于期望最大化可解释的漢字记忆生成 

**Authors**: Jaewook Lee, Alexander Scarlatos, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2507.05137)  

**Abstract**: Learning Japanese vocabulary is a challenge for learners from Roman alphabet backgrounds due to script differences. Japanese combines syllabaries like hiragana with kanji, which are logographic characters of Chinese origin. Kanji are also complicated due to their complexity and volume. Keyword mnemonics are a common strategy to aid memorization, often using the compositional structure of kanji to form vivid associations. Despite recent efforts to use large language models (LLMs) to assist learners, existing methods for LLM-based keyword mnemonic generation function as a black box, offering limited interpretability. We propose a generative framework that explicitly models the mnemonic construction process as driven by a set of common rules, and learn them using a novel Expectation-Maximization-type algorithm. Trained on learner-authored mnemonics from an online platform, our method learns latent structures and compositional rules, enabling interpretable and systematic mnemonics generation. Experiments show that our method performs well in the cold-start setting for new learners while providing insight into the mechanisms behind effective mnemonic creation. 

**Abstract (ZH)**: 罗马字母背景的学习者因书写系统差异而面临日语词汇学习的挑战。日语结合了假名和源自汉字的象形字符 Kanji。Kanji 由于其复杂性和数量而更加复杂。关键词记忆法是一种常见的助记策略，通常利用 Kanji 的构字结构形成生动的联想。尽管最近有努力利用大语言模型（LLMs）辅助学习者，但现有的基于 LLM 的关键词记忆法生成方法缺乏解释性，作为黑盒运作。我们提出一种生成框架，明确地将记忆体构建过程建模为由一套常见规则驱动，并利用一种新型的期望最大化类型算法学习这些规则。在基于一个在线平台的学习者自创记忆法上训练，我们的方法学习潜在结构和构字规则，从而实现可解释和系统的记忆法生成。实验显示，该方法在新学习者冷启动设置中表现良好，同时为有效记忆法创造机制提供了见解。 

---
# An Evaluation of Large Language Models on Text Summarization Tasks Using Prompt Engineering Techniques 

**Title (ZH)**: 使用提示工程技术对大型语言模型在文本摘要任务上的评价 

**Authors**: Walid Mohamed Aly, Taysir Hassan A. Soliman, Amr Mohamed AbdelAziz  

**Link**: [PDF](https://arxiv.org/pdf/2507.05123)  

**Abstract**: Large Language Models (LLMs) continue to advance natural language processing with their ability to generate human-like text across a range of tasks. Despite the remarkable success of LLMs in Natural Language Processing (NLP), their performance in text summarization across various domains and datasets has not been comprehensively evaluated. At the same time, the ability to summarize text effectively without relying on extensive training data has become a crucial bottleneck. To address these issues, we present a systematic evaluation of six LLMs across four datasets: CNN/Daily Mail and NewsRoom (news), SAMSum (dialog), and ArXiv (scientific). By leveraging prompt engineering techniques including zero-shot and in-context learning, our study evaluates the performance using the ROUGE and BERTScore metrics. In addition, a detailed analysis of inference times is conducted to better understand the trade-off between summarization quality and computational efficiency. For Long documents, introduce a sentence-based chunking strategy that enables LLMs with shorter context windows to summarize extended inputs in multiple stages. The findings reveal that while LLMs perform competitively on news and dialog tasks, their performance on long scientific documents improves significantly when aided by chunking strategies. In addition, notable performance variations were observed based on model parameters, dataset properties, and prompt design. These results offer actionable insights into how different LLMs behave across task types, contributing to ongoing research in efficient, instruction-based NLP systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）继续通过其在多种任务中生成人类-like 文本的能力推动自然语言处理的进步。尽管大型语言模型（LLMs）在自然语言处理（NLP）领域取得了显著成功，但它们在各种领域和数据集中的文本摘要性能尚未进行全面评估。同时，有效总结文本而不依赖大量训练数据的能力已成为一个关键瓶颈。为应对这些问题，我们系统性地评估了六种大型语言模型在四个数据集（CNN/Daily Mail和NewsRoom（新闻）、SAMSum（对话）、ArXiv（科技））上的性能。通过利用包括零样本和上下文学习在内的提示工程技术，我们的研究使用ROUGE和BERTScore指标评估性能。此外，我们还详细分析了推理时间，以更好地理解摘要质量和计算效率之间的权衡。对于长文档，引入基于句子的分块策略，使具有较短上下文窗口的语言模型能够分阶段总结扩展输入。研究发现，虽然LLMs在新闻和对话任务上的表现竞争力较强，但在科技长文摘要任务上，通过分块策略的帮助，其性能显著提升。此外，根据模型参数、数据集属性和提示设计，观察到显著的性能差异。这些结果为不同类型的任务提供了关于大语言模型行为的具体洞见，有助于推动高效、基于指令的NLP系统研究。 

---
# LVM4CSI: Enabling Direct Application of Pre-Trained Large Vision Models for Wireless Channel Tasks 

**Title (ZH)**: LVM4CSI: 使预训练大型视觉模型可以直接应用于无线信道任务 

**Authors**: Jiajia Guo, Peiwen Jiang, Chao-Kai Wen, Shi Jin, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05121)  

**Abstract**: Accurate channel state information (CSI) is critical to the performance of wireless communication systems, especially with the increasing scale and complexity introduced by 5G and future 6G technologies. While artificial intelligence (AI) offers a promising approach to CSI acquisition and utilization, existing methods largely depend on task-specific neural networks (NNs) that require expert-driven design and large training datasets, limiting their generalizability and practicality. To address these challenges, we propose LVM4CSI, a general and efficient framework that leverages the structural similarity between CSI and computer vision (CV) data to directly apply large vision models (LVMs) pre-trained on extensive CV datasets to wireless tasks without any fine-tuning, in contrast to large language model-based methods that generally necessitate fine-tuning. LVM4CSI maps CSI tasks to analogous CV tasks, transforms complex-valued CSI into visual formats compatible with LVMs, and integrates lightweight trainable layers to adapt extracted features to specific communication objectives. We validate LVM4CSI through three representative case studies, including channel estimation, human activity recognition, and user localization. Results demonstrate that LVM4CSI achieves comparable or superior performance to task-specific NNs, including an improvement exceeding 9.61 dB in channel estimation and approximately 40% reduction in localization error. Furthermore, it significantly reduces the number of trainable parameters and eliminates the need for task-specific NN design. 

**Abstract (ZH)**: 基于大型视觉模型的无线信道状态信息处理框架 

---
# VerifyLLM: LLM-Based Pre-Execution Task Plan Verification for Robots 

**Title (ZH)**: VerifyLLM：基于LLM的预执行任务计划验证方法 

**Authors**: Danil S. Grigorev, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2507.05118)  

**Abstract**: In the field of robotics, researchers face a critical challenge in ensuring reliable and efficient task planning. Verifying high-level task plans before execution significantly reduces errors and enhance the overall performance of these systems. In this paper, we propose an architecture for automatically verifying high-level task plans before their execution in simulator or real-world environments. Leveraging Large Language Models (LLMs), our approach consists of two key steps: first, the conversion of natural language instructions into Linear Temporal Logic (LTL), followed by a comprehensive analysis of action sequences. The module uses the reasoning capabilities of the LLM to evaluate logical coherence and identify potential gaps in the plan. Rigorous testing on datasets of varying complexity demonstrates the broad applicability of the module to household tasks. We contribute to improving the reliability and efficiency of task planning and addresses the critical need for robust pre-execution verification in autonomous systems. The code is available at this https URL. 

**Abstract (ZH)**: 在机器人领域，研究人员面临确保可靠和高效任务规划的关键挑战。在执行前验证高级任务计划可以显著减少错误并提升这些系统的整体性能。本文提出了一种架构，在模拟器或真实环境中的任务执行前自动验证高级任务计划。该方法利用大型语言模型（LLMs），主要包括两步：首先将自然语言指令转换为线性时序逻辑（LTL），然后对该行动计划序列进行全面分析。模块利用LLM的推理能力评估逻辑连贯性并识别潜在的计划缺口。复杂度各异的数据集上的严格测试表明，该模块适用于家庭任务。本文有助于提高任务规划的可靠性和效率，并满足自主系统在执行前进行 robust 验证的迫切需求。代码见 <https://github.com/XXXXX>。 

---
# Reviving Cultural Heritage: A Novel Approach for Comprehensive Historical Document Restoration 

**Title (ZH)**: 复活文化遗产：一种综合历史文献修复的新方法 

**Authors**: Yuyi Zhang, Peirong Zhang, Zhenhua Yang, Pengyu Yan, Yongxin Shi, Pengwei Liu, Fengjun Guo, Lianwen Jin  

**Link**: [PDF](https://arxiv.org/pdf/2507.05108)  

**Abstract**: Historical documents represent an invaluable cultural heritage, yet have undergone significant degradation over time through tears, water erosion, and oxidation. Existing Historical Document Restoration (HDR) methods primarily focus on single modality or limited-size restoration, failing to meet practical needs. To fill this gap, we present a full-page HDR dataset (FPHDR) and a novel automated HDR solution (AutoHDR). Specifically, FPHDR comprises 1,633 real and 6,543 synthetic images with character-level and line-level locations, as well as character annotations in different damage grades. AutoHDR mimics historians' restoration workflows through a three-stage approach: OCR-assisted damage localization, vision-language context text prediction, and patch autoregressive appearance restoration. The modular architecture of AutoHDR enables seamless human-machine collaboration, allowing for flexible intervention and optimization at each restoration stage. Experiments demonstrate AutoHDR's remarkable performance in HDR. When processing severely damaged documents, our method improves OCR accuracy from 46.83\% to 84.05\%, with further enhancement to 94.25\% through human-machine collaboration. We believe this work represents a significant advancement in automated historical document restoration and contributes substantially to cultural heritage preservation. The model and dataset are available at this https URL. 

**Abstract (ZH)**: 历史文献代表了宝贵的文化遗产，但随着时间的推移，它们经历了严重的退化，受到撕裂、水渍侵蚀和氧化的影响。现有的历史文档修复方法主要侧重于单模态或有限尺寸的修复，无法满足实际需求。为弥补这一缺口，我们提出了一整页历史文档修复数据集（FPHDR）和一种新的自动化历史文档修复解决方案（AutoHDR）。具体来说，FPHDR 包括 1,633 张真实图像和 6,543 张合成图像，其中包含字符级和行级定位以及不同损坏程度的字符注释。AutoHDR 通过三阶段方法模拟历史学家的修复流程：OCR 辅助损坏定位、视觉-语言上下文文本预测以及块自回归外观修复。AutoHDR 的模块化架构使其能够实现无缝的人机协作，在每个修复阶段都允许灵活的干预和优化。实验表明，AutoHDR 在历史文档修复方面表现出色。处理严重损坏的文档时，我们的方法将OCR准确率从46.83%提高到84.05%，并通过人机协作进一步提升至94.25%。我们相信这项工作代表了自动化历史文档修复的重大进展，并对文化遗产保护做出了重要贡献。该模型和数据集可在以下网址获取。 

---
# PRING: Rethinking Protein-Protein Interaction Prediction from Pairs to Graphs 

**Title (ZH)**: PRING: 从成对分析到图表示重新思考蛋白质-蛋白质相互作用预测 

**Authors**: Xinzhe Zheng, Hao Du, Fanding Xu, Jinzhe Li, Zhiyuan Liu, Wenkang Wang, Tao Chen, Wanli Ouyang, Stan Z. Li, Yan Lu, Nanqing Dong, Yang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.05101)  

**Abstract**: Deep learning-based computational methods have achieved promising results in predicting protein-protein interactions (PPIs). However, existing benchmarks predominantly focus on isolated pairwise evaluations, overlooking a model's capability to reconstruct biologically meaningful PPI networks, which is crucial for biology research. To address this gap, we introduce PRING, the first comprehensive benchmark that evaluates protein-protein interaction prediction from a graph-level perspective. PRING curates a high-quality, multi-species PPI network dataset comprising 21,484 proteins and 186,818 interactions, with well-designed strategies to address both data redundancy and leakage. Building on this golden-standard dataset, we establish two complementary evaluation paradigms: (1) topology-oriented tasks, which assess intra and cross-species PPI network construction, and (2) function-oriented tasks, including protein complex pathway prediction, GO module analysis, and essential protein justification. These evaluations not only reflect the model's capability to understand the network topology but also facilitate protein function annotation, biological module detection, and even disease mechanism analysis. Extensive experiments on four representative model categories, consisting of sequence similarity-based, naive sequence-based, protein language model-based, and structure-based approaches, demonstrate that current PPI models have potential limitations in recovering both structural and functional properties of PPI networks, highlighting the gap in supporting real-world biological applications. We believe PRING provides a reliable platform to guide the development of more effective PPI prediction models for the community. The dataset and source code of PRING are available at this https URL. 

**Abstract (ZH)**: 基于深度学习的计算方法在预测蛋白质-蛋白质相互作用（PPIs）方面取得了令人鼓舞的结果。然而，现有的基准主要侧重于单独的成对评估，忽视了模型重建生物意义PPI网络的能力，这对于生物学研究至关重要。为了填补这一空白，我们介绍了PRING，这是第一个从图层面评价蛋白质-蛋白质相互作用预测的综合基准。PRING收集了一个高质量的跨物种PPI网络数据集，包含21,484种蛋白质和186,818种相互作用，并设计了策略来解决数据冗余和泄露问题。基于这个黄金标准数据集，我们建立了两种互补的评估范式：（1）拓扑导向任务，评估跨物种和跨物种PPI网络构建；（2）功能导向任务，包括蛋白质复合体路径预测、GO模块分析和关键蛋白质验证。这些评估不仅反映了模型对网络拓扑的理解能力，还促进了蛋白质功能注释、生物模块检测，甚至疾病机制分析。针对四个代表性模型类别进行了广泛的实验，包括基于序列相似性、基于原始序列、基于蛋白质语言模型和基于结构的方法，结果表明当前的PPI模型在恢复PPI网络的结构和功能特性方面存在潜在局限性，突显了支持实际生物应用场景的差距。我们相信PRING为社区提供了可靠的平台，以指导更有效的PPI预测模型的发展。PRING的数据集和源代码可在以下链接获取：this https URL。 

---
# Beyond Features: How Dataset Design Influences Multi-Agent Trajectory Prediction Performance 

**Title (ZH)**: 超越特征：数据集设计对多agent轨迹预测性能的影响 

**Authors**: Tobias Demmler, Jakob Häringer, Andreas Tamke, Thao Dang, Alexander Hegai, Lars Mikelsons  

**Link**: [PDF](https://arxiv.org/pdf/2507.05098)  

**Abstract**: Accurate trajectory prediction is critical for safe autonomous navigation, yet the impact of dataset design on model performance remains understudied. This work systematically examines how feature selection, cross-dataset transfer, and geographic diversity influence trajectory prediction accuracy in multi-agent settings. We evaluate a state-of-the-art model using our novel L4 Motion Forecasting dataset based on our own data recordings in Germany and the US. This includes enhanced map and agent features. We compare our dataset to the US-centric Argoverse 2 benchmark. First, we find that incorporating supplementary map and agent features unique to our dataset, yields no measurable improvement over baseline features, demonstrating that modern architectures do not need extensive feature sets for optimal performance. The limited features of public datasets are sufficient to capture convoluted interactions without added complexity. Second, we perform cross-dataset experiments to evaluate how effective domain knowledge can be transferred between datasets. Third, we group our dataset by country and check the knowledge transfer between different driving cultures. 

**Abstract (ZH)**: 准确的轨迹预测对于安全的自主导航至关重要，但数据集设计对模型性能的影响仍研究不足。本研究系统地探讨了特征选择、跨数据集迁移和地理多样性如何影响多智能体环境下的轨迹预测精度。我们使用基于德国和美国自身数据记录的新颖L4Motion Forecasting数据集评估了一种最先进的模型，该数据集包含增强的地图和智能体特征。我们将我们的数据集与以美国为中心的Argoverse 2基准进行比较。首先，我们发现将特定于我们数据集的独特地图和智能体特征纳入其中，并未在基线特征上带来可测量的提升，这表明现代架构在最优性能时不需要广泛的特征集。公共数据集的有限特征足以捕捉复杂的交互而不会增加复杂性。其次，我们进行了跨数据集实验，评估领域知识在数据集之间的迁移效果。第三，我们将数据集按国家分组，检查不同驾驶文化之间的知识迁移。 

---
# The Hidden Threat in Plain Text: Attacking RAG Data Loaders 

**Title (ZH)**: 明文中隐含的威胁：攻击RAG数据加载器 

**Authors**: Alberto Castagnaro, Umberto Salviati, Mauro Conti, Luca Pajola, Simeone Pizzi  

**Link**: [PDF](https://arxiv.org/pdf/2507.05093)  

**Abstract**: Large Language Models (LLMs) have transformed human-machine interaction since ChatGPT's 2022 debut, with Retrieval-Augmented Generation (RAG) emerging as a key framework that enhances LLM outputs by integrating external knowledge. However, RAG's reliance on ingesting external documents introduces new vulnerabilities. This paper exposes a critical security gap at the data loading stage, where malicious actors can stealthily corrupt RAG pipelines by exploiting document ingestion.
We propose a taxonomy of 9 knowledge-based poisoning attacks and introduce two novel threat vectors -- Content Obfuscation and Content Injection -- targeting common formats (DOCX, HTML, PDF). Using an automated toolkit implementing 19 stealthy injection techniques, we test five popular data loaders, finding a 74.4% attack success rate across 357 scenarios. We further validate these threats on six end-to-end RAG systems -- including white-box pipelines and black-box services like NotebookLM and OpenAI Assistants -- demonstrating high success rates and critical vulnerabilities that bypass filters and silently compromise output integrity. Our results emphasize the urgent need to secure the document ingestion process in RAG systems against covert content manipulations. 

**Abstract (ZH)**: 大规模语言模型（LLMs）自2022年ChatGPT问世以来重塑了人机交互，检索增强生成（RAG）作为关键框架通过集成外部知识提升了LLM输出。然而，RAG依赖于摄入外部文档引入了新的安全漏洞。本文揭示了数据加载阶段的一个关键安全缺口，恶意行为者可以通过利用文档摄入过程秘密篡改RAG管道。我们提出了9种基于知识的投毒攻击分类，并引入了两种新的威胁向量——内容模糊化和内容注入，针对常见的文件格式（DOCX、HTML、PDF）。使用实现19种隐蔽注入技术的自动化工具包，我们测试了五种流行的数据加载器，在357种情景中取得了74.4%的攻击成功率。进一步在六个端到端的RAG系统上验证这些威胁，包括白盒管道和黑盒服务如NotebookLM和OpenAI助手，展示了高成功率和严重漏洞，可以绕过过滤器并无声地破坏输出完整性。我们的结果强调了迫切需要确保RAG系统中的文档摄入过程免受隐蔽内容操纵。 

---
# Sequential Attention-based Sampling for Histopathological Analysis 

**Title (ZH)**: 基于序列注意力的采样方法用于组织病理学分析 

**Authors**: Tarun G, Naman Malpani, Gugan Thoppe, Sridharan Devarajan  

**Link**: [PDF](https://arxiv.org/pdf/2507.05077)  

**Abstract**: Deep neural networks are increasingly applied for automated histopathology. Yet, whole-slide images (WSIs) are often acquired at gigapixel sizes, rendering it computationally infeasible to analyze them entirely at high resolution. Diagnostic labels are largely available only at the slide-level, because expert annotation of images at a finer (patch) level is both laborious and expensive. Moreover, regions with diagnostic information typically occupy only a small fraction of the WSI, making it inefficient to examine the entire slide at full resolution. Here, we propose SASHA -- {\it S}equential {\it A}ttention-based {\it S}ampling for {\it H}istopathological {\it A}nalysis -- a deep reinforcement learning approach for efficient analysis of histopathological images. First, SASHA learns informative features with a lightweight hierarchical, attention-based multiple instance learning (MIL) model. Second, SASHA samples intelligently and zooms selectively into a small fraction (10-20\%) of high-resolution patches, to achieve reliable diagnosis. We show that SASHA matches state-of-the-art methods that analyze the WSI fully at high-resolution, albeit at a fraction of their computational and memory costs. In addition, it significantly outperforms competing, sparse sampling methods. We propose SASHA as an intelligent sampling model for medical imaging challenges that involve automated diagnosis with exceptionally large images containing sparsely informative features. 

**Abstract (ZH)**: 基于注意力的级联采样方法——SASHA：一种高效的组织病理图像分析深度强化学习方法 

---
# ICAS: Detecting Training Data from Autoregressive Image Generative Models 

**Title (ZH)**: ICAS: 从自回归图像生成模型中检测训练数据 

**Authors**: Hongyao Yu, Yixiang Qiu, Yiheng Yang, Hao Fang, Tianqu Zhuang, Jiaxin Hong, Bin Chen, Hao Wu, Shu-Tao Xia  

**Link**: [PDF](https://arxiv.org/pdf/2507.05068)  

**Abstract**: Autoregressive image generation has witnessed rapid advancements, with prominent models such as scale-wise visual auto-regression pushing the boundaries of visual synthesis. However, these developments also raise significant concerns regarding data privacy and copyright. In response, training data detection has emerged as a critical task for identifying unauthorized data usage in model training. To better understand the vulnerability of autoregressive image generative models to such detection, we conduct the first study applying membership inference to this domain. Our approach comprises two key components: implicit classification and an adaptive score aggregation strategy. First, we compute the implicit token-wise classification score within the query image. Then we propose an adaptive score aggregation strategy to acquire a final score, which places greater emphasis on the tokens with lower scores. A higher final score indicates that the sample is more likely to be involved in the training set. To validate the effectiveness of our method, we adapt existing detection algorithms originally designed for LLMs to visual autoregressive models. Extensive experiments demonstrate the superiority of our method in both class-conditional and text-to-image scenarios. Moreover, our approach exhibits strong robustness and generalization under various data transformations. Furthermore, sufficient experiments suggest two novel key findings: (1) A linear scaling law on membership inference, exposing the vulnerability of large foundation models. (2) Training data from scale-wise visual autoregressive models is easier to detect than other autoregressive this http URL code is available at this https URL. 

**Abstract (ZH)**: 自回归图像生成见证了快速的发展， prominant 模型如规模级视觉自回归推动了视觉合成的边界。然而，这些进展也引发了关于数据隐私和版权的重大关切。为应对这一挑战，训练数据检测已成为关键任务，用于识别未经授权的 数据使用情况。为更好地理解自回归图像生成模型对这种检测的脆弱性，我们首次将成员推理应用于该领域。我们的方法包括两个关键组件：隐式分类和自适应得分聚合策略。首先，我们计算查询图像中的隐式标记级分类得分。然后，我们提出了一种自适应得分聚合策略，以获得最终得分，该策略更重视得分较低的标记。较高的最终得分表明样本更有可能包含在训练集中。为了验证我们方法的有效性，我们将专为LLM设计的现有检测算法适应到视觉自回归模型。大量实验表明，我们的方法在类别条件和文本生成图像情境中都表现优越。此外，我们的方法在各种数据变换下表现出较强的鲁棒性和泛化能力。进一步的实验还揭示了两个新的关键发现：(1) 成员推理的线性标度定律，揭示了大型基础模型的脆弱性。(2) 规模级视觉自回归模型的训练数据比其他自回归模型更容易检测。该研究的代码可在 https://XXX 收到。 

---
# Replacing thinking with tool usage enables reasoning in small language models 

**Title (ZH)**: 用工具替换思考以在小型语言模型中实现推理 

**Authors**: Corrado Rainone, Tim Bakker, Roland Memisevic  

**Link**: [PDF](https://arxiv.org/pdf/2507.05065)  

**Abstract**: Recent advances have established a new machine learning paradigm based on scaling up compute at inference time as well as at training time. In that line of work, a combination of Supervised Fine-Tuning (SFT) on synthetic demonstrations and Reinforcement Learning with Verifiable Rewards (RLVR) is used for training Large Language Models to expend extra compute during inference in the form of "thoughts" expressed in natural language. In this paper, we propose to instead format these tokens as a multi-turn interaction trace with a stateful tool. At each turn, the new state of the tool is appended to the context of the model, whose job is to generate the tokens necessary to control the tool via a custom DSL. We benchmark this approach on the problem of repairing malfunctioning Python code, and show that this constrained setup allows for faster sampling of experience and a denser reward signal, allowing even models of size up to 3B parameters to learn how to proficiently expend additional compute on the task. 

**Abstract (ZH)**: 近期研究确立了一种新的机器学习范式，该范式通过在推理时间和训练时间扩展计算规模来实现。在这一研究方向上，使用监督微调（SFT）合成示例和验证奖励的强化学习（RLVR）来训练大型语言模型，在推理时以自然语言表达的“思考”形式增加额外的计算量。本文提议将这些令牌格式化为具有状态的工具的多轮交互记录。在每次交互中，工具的新状态被附加到模型的上下文中，模型的任务是生成通过自定义DSL控制工具所需的令牌。我们在此问题上对修复功能失常的Python代码进行了基准测试，并展示了这种受限设置允许更快地采样经验并提供更密集的奖励信号，即使是最大的3亿参数模型也能够学会如何有效地在任务中增加额外的计算量。 

---
# INTER: Mitigating Hallucination in Large Vision-Language Models by Interaction Guidance Sampling 

**Title (ZH)**: INTER: 通过交互指导采样减轻大型视觉-语言模型的幻觉问题 

**Authors**: Xin Dong, Shichao Dong, Jin Wang, Jing Huang, Li Zhou, Zenghui Sun, Lihua Jing, Jingsong Lan, Xiaoyong Zhu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.05056)  

**Abstract**: Hallucinations in large vision-language models (LVLMs) pose significant challenges for real-world applications, as LVLMs may generate responses that appear plausible yet remain inconsistent with the associated visual content. This issue rarely occurs in human cognition. We argue that this discrepancy arises from humans' ability to effectively leverage multimodal interaction information in data samples. Specifically, humans typically first gather multimodal information, analyze the interactions across modalities for understanding, and then express their understanding through language. Motivated by this observation, we conduct extensive experiments on popular LVLMs and obtained insights that surprisingly reveal human-like, though less pronounced, cognitive behavior of LVLMs on multimodal samples. Building on these findings, we further propose \textbf{INTER}: \textbf{Inter}action Guidance Sampling, a novel training-free algorithm that mitigate hallucinations without requiring additional data. Specifically, INTER explicitly guides LVLMs to effectively reapply their understanding of multimodal interaction information when generating responses, thereby reducing potential hallucinations. On six benchmarks including VQA and image captioning tasks, INTER achieves an average improvement of up to 3.4\% on five LVLMs compared to the state-of-the-art decoding strategy. The code will be released when the paper is accepted. 

**Abstract (ZH)**: 大型视觉-语言模型中的幻觉对实际应用构成了重大挑战，因为这些模型可能生成看似合理但实际上与关联的视觉内容不一致的响应。这一问题在人类认知中很少出现。我们认为这种差异源于人类能够有效利用数据样本中的多模态交互信息。具体来说，人类通常首先收集多模态信息，分析模态间的交互以进行理解，然后通过语言表达理解。受此观察的启发，我们在流行的大型视觉-语言模型上进行了广泛的实验，并获得了一些令人惊讶的见解，即大型视觉-语言模型在多模态样本上表现出类似人类，尽管不那么明显的认知行为。基于这些发现，我们进一步提出了INTER：交互引导采样，这是一种无需额外数据的创新训练算法，旨在减轻幻觉。具体而言，INTER 显式地指导大型视觉-语言模型在生成响应时有效重新应用对多模态交互信息的理解，从而减少潜在的幻觉。在包括VQA和图像字幕任务在内的六个基准上，INTER 相较于最先进的解码策略，平均提高了多达3.4%。论文被接受后将发布代码。 

---
# Perspectives on How Sociology Can Advance Theorizing about Human-Chatbot Interaction and Developing Chatbots for Social Good 

**Title (ZH)**: 社会学视角下的人机对话理论建构与促进社会福祉聊天机器人的开发 

**Authors**: Celeste Campos-Castillo, Xuan Kang, Linnea I. Laestadius  

**Link**: [PDF](https://arxiv.org/pdf/2507.05030)  

**Abstract**: Recently, research into chatbots (also known as conversational agents, AI agents, voice assistants), which are computer applications using artificial intelligence to mimic human-like conversation, has grown sharply. Despite this growth, sociology lags other disciplines (including computer science, medicine, psychology, and communication) in publishing about chatbots. We suggest sociology can advance understanding of human-chatbot interaction and offer four sociological theories to enhance extant work in this field. The first two theories (resource substitution theory, power-dependence theory) add new insights to existing models of the drivers of chatbot use, which overlook sociological concerns about how social structure (e.g., systemic discrimination, the uneven distribution of resources within networks) inclines individuals to use chatbots, including problematic levels of emotional dependency on chatbots. The second two theories (affect control theory, fundamental cause of disease theory) help inform the development of chatbot-driven interventions that minimize safety risks and enhance equity by leveraging sociological insights into how chatbot outputs could attend to cultural contexts (e.g., affective norms) to promote wellbeing and enhance communities (e.g., opportunities for civic participation). We discuss the value of applying sociological theories for advancing theorizing about human-chatbot interaction and developing chatbots for social good. 

**Abstract (ZH)**: 最近，关于聊天机器人的研究（也称为对话代理、AI代理、语音助手）呈快速增长之势，这些是使用人工智能模仿人类对话的计算机应用程序。尽管如此， sociology在关于聊天机器人的出版物方面仍然落后于计算机科学、医学、心理学和传播学等其他学科。我们建议 sociology可以通过提出四条社会学理论来推进对人类与聊天机器人交互的理解，并增强该领域现有的研究工作。前两条理论（资源替代理论、权力依赖理论）为聊天机器人使用动因的现有模型增添了新的见解，这些模型忽视了社会结构（如系统性歧视、网络内部资源分配不均）如何促使个人使用聊天机器人，包括对聊天机器人的情感依赖程度可能达到不健康的水平的社会学担忧。后两条理论（情感控制理论、疾病根本原因理论）有助于指导由聊天机器人驱动的干预措施的发展，以减轻安全风险并促进公平，这些干预措施利用社会学洞见来关注聊天机器人输出如何考虑到文化背景（如情感规范）促进福祉并增强社区（如公民参与的机会）。我们探讨了应用社会学理论以推进对人类与聊天机器人交互的理解以及开发用于社会福利的聊天机器人的价值。 

---
# Adaptation of Multi-modal Representation Models for Multi-task Surgical Computer Vision 

**Title (ZH)**: 多模态表示模型在多任务手术计算机视觉中的适应性研究 

**Authors**: Soham Walimbe, Britty Baby, Vinkle Srivastav, Nicolas Padoy  

**Link**: [PDF](https://arxiv.org/pdf/2507.05020)  

**Abstract**: Surgical AI often involves multiple tasks within a single procedure, like phase recognition or assessing the Critical View of Safety in laparoscopic cholecystectomy. Traditional models, built for one task at a time, lack flexibility, requiring a separate model for each. To address this, we introduce MML-SurgAdapt, a unified multi-task framework with Vision-Language Models (VLMs), specifically CLIP, to handle diverse surgical tasks through natural language supervision. A key challenge in multi-task learning is the presence of partial annotations when integrating different tasks. To overcome this, we employ Single Positive Multi-Label (SPML) learning, which traditionally reduces annotation burden by training models with only one positive label per instance. Our framework extends this approach to integrate data from multiple surgical tasks within a single procedure, enabling effective learning despite incomplete or noisy annotations. We demonstrate the effectiveness of our model on a combined dataset consisting of Cholec80, Endoscapes2023, and CholecT50, utilizing custom prompts. Extensive evaluation shows that MML-SurgAdapt performs comparably to task-specific benchmarks, with the added advantage of handling noisy annotations. It also outperforms the existing SPML frameworks for the task. By reducing the required labels by 23%, our approach proposes a more scalable and efficient labeling process, significantly easing the annotation burden on clinicians. To our knowledge, this is the first application of SPML to integrate data from multiple surgical tasks, presenting a novel and generalizable solution for multi-task learning in surgical computer vision. Implementation is available at: this https URL 

**Abstract (ZH)**: Surgical AI often involves multiple tasks within a single procedure, such as phase recognition or assessing the Critical View of Safety in laparoscopic cholecystectomy. Traditional models, designed for one task at a time, lack flexibility, requiring separate models for each. To address this, we introduce MML-SurgAdapt, a unified multi-task framework using Vision-Language Models (VLMs), specifically CLIP, to handle various surgical tasks through natural language supervision. A key challenge in multi-task learning is the presence of partial annotations when integrating different tasks. To overcome this, we employ Single Positive Multi-Label (SPML) learning, traditionally reducing annotation burden by training models with only one positive label per instance. Our framework extends this approach to integrate data from multiple surgical tasks within a single procedure, enabling effective learning despite incomplete or noisy annotations. We demonstrate the effectiveness of our model on a combined dataset consisting of Cholec80, Endoscapes2023, and CholecT50, utilizing custom prompts. Extensive evaluation shows that MML-SurgAdapt performs comparably to task-specific benchmarks, with the added advantage of handling noisy annotations. It also outperforms existing SPML frameworks for the task. By reducing the required labels by 23%, our approach proposes a more scalable and efficient labeling process, significantly easing the annotation burden on clinicians. To our knowledge, this is the first application of SPML to integrate data from multiple surgical tasks, presenting a novel and generalizable solution for multi-task learning in surgical computer vision. Implementation is available at: this https URL. 

---
# Meta-Learning Transformers to Improve In-Context Generalization 

**Title (ZH)**: 元学习变换器以提高上下文内泛化能力 

**Authors**: Lorenzo Braccaioli, Anna Vettoruzzo, Prabhant Singh, Joaquin Vanschoren, Mohamed-Rafik Bouguelia, Nicola Conci  

**Link**: [PDF](https://arxiv.org/pdf/2507.05019)  

**Abstract**: In-context learning enables transformer models to generalize to new tasks based solely on input prompts, without any need for weight updates. However, existing training paradigms typically rely on large, unstructured datasets that are costly to store, difficult to evaluate for quality and balance, and pose privacy and ethical concerns due to the inclusion of sensitive information. Motivated by these limitations and risks, we propose an alternative training strategy where we leverage a collection of multiple, small-scale, and domain-specific datasets. We empirically demonstrate that the increased quality and diversity of such data improve the generalization abilities of in-context learners beyond their training domain, while achieving comparable performance with models trained on a single large-scale dataset. We investigate this paradigm by leveraging meta-learning to train an in-context learner on the Meta-Album collection under several settings. Firstly, we show the performance in a controlled environment, where the test domain is completely excluded from the training knowledge. Secondly, we explore the robustness of these models to forgetting in a continual scenario where the information is accessible for a limited time. Finally, we explore the more challenging unsupervised scenario. Our findings demonstrate that transformers still generalize for in-context prediction when trained on a curated dataset collection while offering advantages in modularity and replaceability. 

**Abstract (ZH)**: 基于上下文学习透过输入提示使变压器模型能够在无需权重更新的情况下泛化到新任务。然而，现有的训练范式通常依赖于大规模、无结构的数据集，这些数据集存储成本高、质量评估困难、平衡性差，并且由于包含了敏感信息而带来隐私和伦理方面的担忧。鉴于这些限制和风险，我们提出了一种替代的训练策略，其中我们利用多个小型且领域特定的数据集。我们实验证明，此类数据的质量和多样性提高有助于在训练领域之外增强基于上下文的学习器的泛化能力，同时在性能上与在单一大规模数据集上训练的模型相当。我们通过元学习，将基于上下文的学习器在Meta-Album集合下进行了训练，以探索这一范式。首先，我们在一个受控环境中展示了性能，其中测试领域完全不包括训练知识。其次，我们在持续学习场景中探究了这些模型对抗遗忘的鲁棒性，信息在此场景下仅在有限时间内可用。最后，我们探讨了更具挑战性的无监督场景。我们的研究结果表明，尽管是在精心筛选的数据集上进行训练，变压器模型仍然能在基于上下文的预测中泛化，并且具备模块化和可替换的优点。 

---
# Multi-modal Representations for Fine-grained Multi-label Critical View of Safety Recognition 

**Title (ZH)**: 多模态表示在细粒度多标签安全关键视角识别中的应用 

**Authors**: Britty Baby, Vinkle Srivastav, Pooja P. Jain, Kun Yuan, Pietro Mascagni, Nicolas Padoy  

**Link**: [PDF](https://arxiv.org/pdf/2507.05007)  

**Abstract**: The Critical View of Safety (CVS) is crucial for safe laparoscopic cholecystectomy, yet assessing CVS criteria remains a complex and challenging task, even for experts. Traditional models for CVS recognition depend on vision-only models learning with costly, labor-intensive spatial annotations. This study investigates how text can be harnessed as a powerful tool for both training and inference in multi-modal surgical foundation models to automate CVS recognition. Unlike many existing multi-modal models, which are primarily adapted for multi-class classification, CVS recognition requires a multi-label framework. Zero-shot evaluation of existing multi-modal surgical models shows a significant performance gap for this task. To address this, we propose CVS-AdaptNet, a multi-label adaptation strategy that enhances fine-grained, binary classification across multiple labels by aligning image embeddings with textual descriptions of each CVS criterion using positive and negative prompts. By adapting PeskaVLP, a state-of-the-art surgical foundation model, on the Endoscapes-CVS201 dataset, CVS-AdaptNet achieves 57.6 mAP, improving over the ResNet50 image-only baseline (51.5 mAP) by 6 points. Our results show that CVS-AdaptNet's multi-label, multi-modal framework, enhanced by textual prompts, boosts CVS recognition over image-only methods. We also propose text-specific inference methods, that helps in analysing the image-text alignment. While further work is needed to match state-of-the-art spatial annotation-based methods, this approach highlights the potential of adapting generalist models to specialized surgical tasks. Code: this https URL 

**Abstract (ZH)**: 基于文本的多模态手术基础模型在安全评估中的应用：CVS-AdaptNet研究 

---
# Classification of autoimmune diseases from Peripheral blood TCR repertoires by multimodal multi-instance learning 

**Title (ZH)**: 基于多模态多实例学习的外周血TCR repertoire在自身免疫疾病分类中的应用 

**Authors**: Ruihao Zhang, Fei Ye, Dandan Meng, Yixuan Huang, Maochen, Xiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04981)  

**Abstract**: T cell receptor (TCR) repertoires encode critical immunological signatures for autoimmune diseases, yet their clinical application remains limited by sequence sparsity and low witness rates. We developed EAMil, a multi-instance deep learning framework that leverages TCR sequencing data to diagnose systemic lupus erythematosus (SLE) and rheumatoid arthritis (RA) with exceptional accuracy. By integrating PrimeSeq feature extraction with ESMonehot encoding and enhanced gate attention mechanisms, our model achieved state-of-the-art performance with AUCs of 98.95% for SLE and 97.76% for RA. EAMil successfully identified disease-associated genes with over 90% concordance with established differential analyses and effectively distinguished disease-specific TCR genes. The model demonstrated robustness in classifying multiple disease categories, utilizing the SLEDAI score to stratify SLE patients by disease severity as well as to diagnose the site of damage in SLE patients, and effectively controlling for confounding factors such as age and gender. This interpretable framework for immune receptor analysis provides new insights for autoimmune disease detection and classification with broad potential clinical applications across immune-mediated conditions. 

**Abstract (ZH)**: T细胞受体(TCR) repertoire编码了自身免疫性疾病的关键免疫学标志，但由于序列稀疏性和低检测率，其临床应用受到限制。我们开发了一种名为EAMil的多实例深度学习框架，利用TCR测序数据以极高的准确性诊断系统性红斑狼疮(SLE)和类风湿性关节炎(RA)。通过集成PrimeSeq特征提取、ESMon-hot编码和增强门控注意力机制，我们的模型在SLE上的AUC达到98.95%，在RA上的AUC达到97.76%，实现了最先进的性能。EAMil成功地识别了与疾病相关的基因，与现有差异分析的一致性超过90%，有效地区分了疾病的特异性TCR基因。该模型在分类多种疾病类别时表现出鲁棒性，使用SLEDAI评分按疾病严重程度对SLE患者进行分层，并诊断SLE患者的受损部位，有效地控制了年龄和性别等混杂因素。该可解释的免疫受体分析框架为自身免疫疾病的检测和分类提供了新的见解，并具有广泛的临床应用潜力，适用于免疫介导的疾病。 

---
# LAPS-Diff: A Diffusion-Based Framework for Singing Voice Synthesis With Language Aware Prosody-Style Guided Learning 

**Title (ZH)**: LAPS-Diff：一种基于扩散的歌声合成框架，带有语言意识音调风格引导学习 

**Authors**: Sandipan Dhar, Mayank Gupta, Preeti Rao  

**Link**: [PDF](https://arxiv.org/pdf/2507.04966)  

**Abstract**: The field of Singing Voice Synthesis (SVS) has seen significant advancements in recent years due to the rapid progress of diffusion-based approaches. However, capturing vocal style, genre-specific pitch inflections, and language-dependent characteristics remains challenging, particularly in low-resource scenarios. To address this, we propose LAPS-Diff, a diffusion model integrated with language-aware embeddings and a vocal-style guided learning mechanism, specifically designed for Bollywood Hindi singing style. We curate a Hindi SVS dataset and leverage pre-trained language models to extract word and phone-level embeddings for an enriched lyrics representation. Additionally, we incorporated a style encoder and a pitch extraction model to compute style and pitch losses, capturing features essential to the naturalness and expressiveness of the synthesized singing, particularly in terms of vocal style and pitch variations. Furthermore, we utilize MERT and IndicWav2Vec models to extract musical and contextual embeddings, serving as conditional priors to refine the acoustic feature generation process further. Based on objective and subjective evaluations, we demonstrate that LAPS-Diff significantly improves the quality of the generated samples compared to the considered state-of-the-art (SOTA) model for our constrained dataset that is typical of the low resource scenario. 

**Abstract (ZH)**: 基于语言感知嵌入和声乐风格引导学习机制的LAPS-Diff声乐语音合成模型 

---
# Hear-Your-Click: Interactive Video-to-Audio Generation via Object-aware Contrastive Audio-Visual Fine-tuning 

**Title (ZH)**: 听你的点击：基于对象aware对比音频-视觉微调的交互式视频到音频生成 

**Authors**: Yingshan Liang, Keyu Fan, Zhicheng Du, Yiran Wang, Qingyang Shi, Xinyu Zhang, Jiasheng Lu, Peiwu Qin  

**Link**: [PDF](https://arxiv.org/pdf/2507.04959)  

**Abstract**: Video-to-audio (V2A) generation shows great potential in fields such as film production. Despite significant advances, current V2A methods, which rely on global video information, struggle with complex scenes and often fail to generate audio tailored to specific objects or regions in the videos. To address these limitations, we introduce Hear-Your-Click, an interactive V2A framework that enables users to generate sounds for specific objects in the videos by simply clicking on the frame. To achieve this, we propose Object-aware Contrastive Audio-Visual Fine-tuning (OCAV) with a Mask-guided Visual Encoder (MVE) to obtain object-level visual features aligned with corresponding audio segments. Furthermore, we tailor two data augmentation strategies: Random Video Stitching (RVS) and Mask-guided Loudness Modulation (MLM), aimed at enhancing the model's sensitivity to the segmented objects. To effectively measure the audio-visual correspondence, we design a new evaluation metric, the CAV score, for evaluation. Extensive experiments demonstrate that our framework offers more precise control and improved generation performance across various metrics. Project Page: this https URL 

**Abstract (ZH)**: 视频到音频（V2A）生成在电影制作等领域展现出巨大的潜力。尽管取得了显著进展，当前依赖全局视频信息的V2A方法在处理复杂场景时常常无法生成针对视频中特定对象或区域的定制化音频。为解决这些局限性，我们引入了Hear-Your-Click，这是一个交互式的V2A框架，允许用户通过单击帧来生成视频中特定对象的声音。为此，我们提出了对象感知对比音频-视觉微调（OCAV）和掩码引导视觉编码器（MVE），以获得与相应音频段对齐的对象级视觉特征。此外，我们定制了两种数据增强策略：随机视频拼接（RVS）和掩码引导响度调制（MLM），旨在提高模型对分割对象的敏感度。为了有效衡量音频-视觉对应关系，我们设计了一种新的评估指标——CAV分数。广泛的实验表明，我们的框架在多种指标上提供了更精确的控制和改进的生成性能。项目页面: this https URL。 

---
# EXPOTION: Facial Expression and Motion Control for Multimodal Music Generation 

**Title (ZH)**: EXPOTION：多模态音乐生成中的面部表情和动作控制 

**Authors**: Fathinah Izzati, Xinyue Li, Gus Xia  

**Link**: [PDF](https://arxiv.org/pdf/2507.04955)  

**Abstract**: We propose Expotion (Facial Expression and Motion Control for Multimodal Music Generation), a generative model leveraging multimodal visual controls - specifically, human facial expressions and upper-body motion - as well as text prompts to produce expressive and temporally accurate music. We adopt parameter-efficient fine-tuning (PEFT) on the pretrained text-to-music generation model, enabling fine-grained adaptation to the multimodal controls using a small dataset. To ensure precise synchronization between video and music, we introduce a temporal smoothing strategy to align multiple modalities. Experiments demonstrate that integrating visual features alongside textual descriptions enhances the overall quality of generated music in terms of musicality, creativity, beat-tempo consistency, temporal alignment with the video, and text adherence, surpassing both proposed baselines and existing state-of-the-art video-to-music generation models. Additionally, we introduce a novel dataset consisting of 7 hours of synchronized video recordings capturing expressive facial and upper-body gestures aligned with corresponding music, providing significant potential for future research in multimodal and interactive music generation. 

**Abstract (ZH)**: Expotion：基于多模态视觉控制的音乐生成模型 

---
# DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer 

**Title (ZH)**: DC-AR：高效掩码自回归图像生成的深度压缩混合分词器 

**Authors**: Yecheng Wu, Junyu Chen, Zhuoyang Zhang, Enze Xie, Jincheng Yu, Junsong Chen, Jinyi Hu, Yao Lu, Song Han, Han Cai  

**Link**: [PDF](https://arxiv.org/pdf/2507.04947)  

**Abstract**: We introduce DC-AR, a novel masked autoregressive (AR) text-to-image generation framework that delivers superior image generation quality with exceptional computational efficiency. Due to the tokenizers' limitations, prior masked AR models have lagged behind diffusion models in terms of quality or efficiency. We overcome this limitation by introducing DC-HT - a deep compression hybrid tokenizer for AR models that achieves a 32x spatial compression ratio while maintaining high reconstruction fidelity and cross-resolution generalization ability. Building upon DC-HT, we extend MaskGIT and create a new hybrid masked autoregressive image generation framework that first produces the structural elements through discrete tokens and then applies refinements via residual tokens. DC-AR achieves state-of-the-art results with a gFID of 5.49 on MJHQ-30K and an overall score of 0.69 on GenEval, while offering 1.5-7.9x higher throughput and 2.0-3.5x lower latency compared to prior leading diffusion and autoregressive models. 

**Abstract (ZH)**: DC-AR：一种新型掩码自回归文本到图像生成框架 

---
# Object-centric Denoising Diffusion Models for Physical Reasoning 

**Title (ZH)**: 以物为中心的去噪扩散模型用于物理推理 

**Authors**: Moritz Lange, Raphael C. Engelhardt, Wolfgang Konen, Andrew Melnik, Laurenz Wiskott  

**Link**: [PDF](https://arxiv.org/pdf/2507.04920)  

**Abstract**: Reasoning about the trajectories of multiple, interacting objects is integral to physical reasoning tasks in machine learning. This involves conditions imposed on the objects at different time steps, for instance initial states or desired goal states. Existing approaches in physical reasoning generally rely on autoregressive modeling, which can only be conditioned on initial states, but not on later states. In fields such as planning for reinforcement learning, similar challenges are being addressed with denoising diffusion models. In this work, we propose an object-centric denoising diffusion model architecture for physical reasoning that is translation equivariant over time, permutation equivariant over objects, and can be conditioned on arbitrary time steps for arbitrary objects. We demonstrate how this model can solve tasks with multiple conditions and examine its performance when changing object numbers and trajectory lengths during inference. 

**Abstract (ZH)**: 关于多个相互作用物体的轨迹推理是机器学习中物理推理任务的核心。现有的物理推理方法通常依赖于自回归建模，只能条件化初始状态，而不能条件化后续状态。在强化学习规划等领域，类似的挑战正通过去噪扩散模型进行解决。在本工作中，我们提出了一种以物体为中心的去噪扩散模型架构，该架构在时间上是平移不变的，在物体上是置换不变的，并且可以任意时间步长对任意物体进行条件化。我们展示了该模型如何解决具有多个条件的任务，并评估了在推理过程中改变物体数量和轨迹长度时的性能。 

---
# Leadership Detection via Time-Lagged Correlation-Based Network Inference 

**Title (ZH)**: 基于时间延迟相关性网络推断的领导人检测 

**Authors**: Thayanne França da Silva, José Everardo Bessa Maia  

**Link**: [PDF](https://arxiv.org/pdf/2507.04917)  

**Abstract**: Understanding leadership dynamics in collective behavior is a key challenge in animal ecology, swarm robotics, and intelligent transportation. Traditional information-theoretic approaches, including Transfer Entropy (TE) and Time-Lagged Mutual Information (TLMI), have been widely used to infer leader-follower relationships but face critical limitations in noisy or short-duration datasets due to their reliance on robust probability estimations. This study proposes a method based on dynamic network inference using time-lagged correlations across multiple kinematic variables: velocity, acceleration, and direction. Our approach constructs directed influence graphs over time, enabling the identification of leadership patterns without the need for large volumes of data or parameter-sensitive discretization. We validate our method through two multi-agent simulations in NetLogo: a modified Vicsek model with informed leaders and a predator-prey model featuring coordinated and independent wolf groups. Experimental results demonstrate that the network-based method outperforms TE and TLMI in scenarios with limited spatiotemporal observations, ranking true leaders at the top of influence metrics more consistently than TE and TLMI. 

**Abstract (ZH)**: 理解群体行为中的领导动态是动物生态学、 swarm 机器人技术和智能交通系统中的一个关键挑战。传统的信息理论方法，包括转移信息熵（TE）和时间延迟互信息（TLMI），广泛用于推断领导者-跟随者关系，但在嘈杂或短暂持续时间的数据集中由于依赖稳健的概率估计而面临关键限制。本研究提出了一种基于动态网络推断的方法，该方法使用多个动态变量（速度、加速度和方向）的时间延迟相关性来构建跨时间的有向影响图，无需大量数据或参数敏感的离散化即可识别领导模式。通过NetLogo中的两个多智能体模拟（修改后的Vicsek模型和具有协调和独立狼群的捕食者-被捕食者模型）验证了该方法。实验结果表明，在时空观测有限的场景中，网络方法在影响力指标中更一致地排名真实领导者，优于TE和TLMI。 

---
# HV-MMBench: Benchmarking MLLMs for Human-Centric Video Understanding 

**Title (ZH)**: HV-MMBench: 人类中心视频理解的MLLMs基准测试 

**Authors**: Yuxuan Cai, Jiangning Zhang, Zhenye Gan, Qingdong He, Xiaobin Hu, Junwei Zhu, Yabiao Wang, Chengjie Wang, Zhucun Xue, Xinwei He, Xiang Bai  

**Link**: [PDF](https://arxiv.org/pdf/2507.04909)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated significant advances in visual understanding tasks involving both images and videos. However, their capacity to comprehend human-centric video data remains underexplored, primarily due to the absence of comprehensive and high-quality evaluation benchmarks. Existing human-centric benchmarks predominantly emphasize video generation quality and action recognition, while overlooking essential perceptual and cognitive abilities required in human-centered scenarios. Furthermore, they are often limited by single-question paradigms and overly simplistic evaluation metrics. To address above limitations, we propose a modern HV-MMBench, a rigorously curated benchmark designed to provide a more holistic evaluation of MLLMs in human-centric video understanding. Compared to existing human-centric video benchmarks, our work offers the following key features: (1) Diverse evaluation dimensions: HV-MMBench encompasses 15 tasks, ranging from basic attribute perception (e.g., age estimation, emotion recognition) to advanced cognitive reasoning (e.g., social relationship prediction, intention prediction), enabling comprehensive assessment of model capabilities; (2) Varied data types: The benchmark includes multiple-choice, fill-in-blank, true/false, and open-ended question formats, combined with diverse evaluation metrics, to more accurately and robustly reflect model performance; (3) Multi-domain video coverage: The benchmark spans 50 distinct visual scenarios, enabling comprehensive evaluation across fine-grained scene variations; (4) Temporal coverage: The benchmark covers videos from short-term (10 seconds) to long-term (up to 30min) durations, supporting systematic analysis of models temporal reasoning abilities across diverse contextual lengths. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在涉及图像和视频的视觉理解任务中取得了显著进展。然而，它们对以人类为中心的视频数据的理解能力尚未得到充分探索，主要是因为缺乏全面且高质量的评估基准。现有的以人类为中心的基准主要侧重于视频生成质量和动作识别，而忽略了人类中心场景中所需的感知和认知能力。此外，它们往往受限于单一问题范式和过于简化的评估指标。为解决上述局限性，我们提出了一项现代HV-MMBench基准，旨在为MLLMs在以人类为中心的视频理解中的综合评估提供更严谨的做法。与现有以人类为中心的视频基准相比，我们工作具有以下关键特点：（1）多维度评估：HV-MMBench包含15项任务，从基本属性感知（如年龄 estimation, 情绪识别）到高级认知推理（如社会关系预测, 意图预测），以全面评估模型能力；（2）多样化的数据类型：基准包括多项选择题、填空题、是非题和开放型问题格式，并结合多种评估指标，以更准确和稳健地反映模型性能；（3）多领域视频覆盖：基准涵盖了50种不同的视觉场景，以全面评估细粒度场景变化；（4）时间覆盖：基准涵盖了从短时（10秒）到长时（最大30分钟）的视频，支持对模型跨不同上下文长度时间推理能力的系统分析。 

---
# BackFed: An Efficient & Standardized Benchmark Suite for Backdoor Attacks in Federated Learning 

**Title (ZH)**: BackFed: 一种高效的联邦学习后门攻击标准化基准套件 

**Authors**: Thinh Dao, Dung Thuy Nguyen, Khoa D Doan, Kok-Seng Wong  

**Link**: [PDF](https://arxiv.org/pdf/2507.04903)  

**Abstract**: Federated Learning (FL) systems are vulnerable to backdoor attacks, where adversaries train their local models on poisoned data and submit poisoned model updates to compromise the global model. Despite numerous proposed attacks and defenses, divergent experimental settings, implementation errors, and unrealistic assumptions hinder fair comparisons and valid conclusions about their effectiveness in real-world scenarios. To address this, we introduce BackFed - a comprehensive benchmark suite designed to standardize, streamline, and reliably evaluate backdoor attacks and defenses in FL, with a focus on practical constraints. Our benchmark offers key advantages through its multi-processing implementation that significantly accelerates experimentation and the modular design that enables seamless integration of new methods via well-defined APIs. With a standardized evaluation pipeline, we envision BackFed as a plug-and-play environment for researchers to comprehensively and reliably evaluate new attacks and defenses. Using BackFed, we conduct large-scale studies of representative backdoor attacks and defenses across both Computer Vision and Natural Language Processing tasks with diverse model architectures and experimental settings. Our experiments critically assess the performance of proposed attacks and defenses, revealing unknown limitations and modes of failures under practical conditions. These empirical insights provide valuable guidance for the development of new methods and for enhancing the security of FL systems. Our framework is openly available at this https URL. 

**Abstract (ZH)**: 联邦学习（FL）系统易受后门攻击，攻击者可以在中毒数据上训练其本地模型，并提交中毒模型更新以 compromize 全局模型。尽管提出了许多攻击和防御方法，但由于实验设置差异、实现错误以及不切实际的假设，阻碍了在实际场景中公平比较和得出有效结论。为了解决这个问题，我们引入了BackFed - 一个旨在标准化、简化并可靠评估FL中后门攻击和防御的基准套件，重点关注实际约束条件。我们的基准通过多处理实现显著加速了实验，并通过模块化设计支持通过明确定义的API无缝集成新方法。通过标准化评估流程，我们设想BackFed为研究人员提供了一个即插即用的环境，以全面可靠地评估新的攻击和防御方法。使用BackFed，我们针对计算机视觉和自然语言处理任务进行了广泛的代表性后门攻击和防御的大规模研究，涉及多样化的模型架构和实验设置。我们的实验严格评估了所提出的攻击和防御方法，在实际条件下的性能揭示了未知的局限性和失败模式。这些经验见解为新方法的开发以及增强FL系统的安全性提供了宝贵的指导。我们的框架已公开可用，详情请访问 this https URL。 

---
# Emergent Semantics Beyond Token Embeddings: Transformer LMs with Frozen Visual Unicode Representations 

**Title (ZH)**: 超越令牌嵌入的新兴语义：冻结视觉Unicode表示的变换器语言模型 

**Authors**: A. Bochkov  

**Link**: [PDF](https://arxiv.org/pdf/2507.04886)  

**Abstract**: Understanding the locus of semantic representation in large language models (LLMs) is crucial for interpretability and architectural innovation. The dominant paradigm posits that trainable input embeddings serve as foundational "meaning vectors." This paper challenges that view. We construct Transformer models where the embedding layer is entirely frozen, with vectors derived not from data, but from the visual structure of Unicode glyphs. These non-semantic, precomputed visual embeddings are fixed throughout training. Our method is compatible with any tokenizer, including a novel Unicode-centric tokenizer we introduce to ensure universal text coverage. Despite the absence of trainable, semantically initialized embeddings, our models converge, generate coherent text, and, critically, outperform architecturally identical models with trainable embeddings on the MMLU reasoning benchmark. We attribute this to "representational interference" in conventional models, where the embedding layer is burdened with learning both structural and semantic features. Our results indicate that high-level semantics are not inherent to input embeddings but are an emergent property of the Transformer's compositional architecture and data scale. This reframes the role of embeddings from meaning containers to structural primitives. We release all code and models to foster further research. 

**Abstract (ZH)**: 理解大型语言模型中语义表示的位置对于可解释性和架构创新至关重要。当前的主导观点认为可训练的输入嵌入充当基础的“意义向量”。本文挑战这一观点。我们构建了Transformer模型，其中嵌入层完全冻结，嵌入向量并非来源于数据，而是来源于Unicode字符符号的视觉结构。这些非语义的、预先计算的视觉嵌入在整个训练过程中保持不变。该方法适用于任何分词器，包括我们引入的一种新的以Unicode为中心的分词器，以确保文本的全面覆盖。尽管缺乏可训练的初始化语义嵌入，我们的模型仍能收敛，生成连贯的文本，并且在MMLU推理基准测试中，我们的模型在架构上与具有可训练嵌入的模型相比表现更优。我们归因于此种传统的模型中“表示干扰”，其中嵌入层需要学习结构和语义特征。我们的结果显示，高层语义并非输入嵌入的固有特征，而是Transformer组合架构和数据规模的 emergent 属性。这重新定义了嵌入的角色，从意义容器转变为结构基本元素。我们已发布所有代码和模型，以促进进一步的研究。 

---
# Beyond Training-time Poisoning: Component-level and Post-training Backdoors in Deep Reinforcement Learning 

**Title (ZH)**: 超越训练时污染：深度强化学习中的组件级和后训练后门 

**Authors**: Sanyam Vyas, Alberto Caron, Chris Hicks, Pete Burnap, Vasilios Mavroudis  

**Link**: [PDF](https://arxiv.org/pdf/2507.04883)  

**Abstract**: Deep Reinforcement Learning (DRL) systems are increasingly used in safety-critical applications, yet their security remains severely underexplored. This work investigates backdoor attacks, which implant hidden triggers that cause malicious actions only when specific inputs appear in the observation space. Existing DRL backdoor research focuses solely on training-time attacks requiring unrealistic access to the training pipeline. In contrast, we reveal critical vulnerabilities across the DRL supply chain where backdoors can be embedded with significantly reduced adversarial privileges. We introduce two novel attacks: (1) TrojanentRL, which exploits component-level flaws to implant a persistent backdoor that survives full model retraining; and (2) InfrectroRL, a post-training backdoor attack which requires no access to training, validation, nor test data. Empirical and analytical evaluations across six Atari environments show our attacks rival state-of-the-art training-time backdoor attacks while operating under much stricter adversarial constraints. We also demonstrate that InfrectroRL further evades two leading DRL backdoor defenses. These findings challenge the current research focus and highlight the urgent need for robust defenses. 

**Abstract (ZH)**: 深度 reinforcement learning (DRL) 系统在安全关键应用中的使用日益增多，但其安全性仍严重未被探索。本研究调查了后门攻击，这种攻击植入隐式触发器，只有在观测空间出现特定输入时才会导致恶意行为。现有的 DRL 后门研究仅关注训练时间攻击，需要不现实的访问训练管道的权限。相比之下，我们揭示了 DRL 供应链中的关键漏洞，在这些漏洞中，后门可以被嵌入，且敌手权限明显降低。我们提出了两个新型攻击：（1）TrojanentRL，利用组件级漏洞植入持久性后门，即使进行全模型重新训练也能存活；（2）InfrectroRL，一种后训练后门攻击，无需访问训练、验证或测试数据。在六个 Atari 环境上的实证和分析评估显示，我们的攻击在严格的敌手约束条件下，与最先进的训练时间后门攻击性能相当。我们还展示了 InfrectroRL 进一步规避了两种领先的 DRL 后门防御措施。这些发现挑战了当前的研究重点，并强调了迫切需要稳健的防御措施。 

---
# HGNet: High-Order Spatial Awareness Hypergraph and Multi-Scale Context Attention Network for Colorectal Polyp Detection 

**Title (ZH)**: HGNet：高阶空间意识超图和多尺度上下文注意力网络在结肠息肉检测中的应用 

**Authors**: Xiaofang Liu, Lingling Sun, Xuqing Zhang, Yuannong Ye, Bin zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.04880)  

**Abstract**: Colorectal cancer (CRC) is closely linked to the malignant transformation of colorectal polyps, making early detection essential. However, current models struggle with detecting small lesions, accurately localizing boundaries, and providing interpretable decisions. To address these issues, we propose HGNet, which integrates High-Order Spatial Awareness Hypergraph and Multi-Scale Context Attention. Key innovations include: (1) an Efficient Multi-Scale Context Attention (EMCA) module to enhance lesion feature representation and boundary modeling; (2) the deployment of a spatial hypergraph convolution module before the detection head to capture higher-order spatial relationships between nodes; (3) the application of transfer learning to address the scarcity of medical image data; and (4) Eigen Class Activation Map (Eigen-CAM) for decision visualization. Experimental results show that HGNet achieves 94% accuracy, 90.6% recall, and 90% mAP@0.5, significantly improving small lesion differentiation and clinical interpretability. The source code will be made publicly available upon publication of this paper. 

**Abstract (ZH)**: 结直肠癌（CRC）与结直肠息肉的恶性转化密切相关，早期检测至关重要。然而，现有模型在检测小病灶、准确定位边界以及提供可解释的决策方面存在困难。为解决这些问题，我们提出了HGNet，它结合了高阶空间意识超图和多尺度上下文注意机制。关键创新包括：(1) 一种高效的多尺度上下文注意模块（EMCA），以增强病灶特征表示和边界建模；(2) 在检测头之前部署空间超图卷积模块，以捕获节点之间的高阶空间关系；(3) 应用迁移学习以解决医学图像数据稀缺性问题；(4) 使用Eigen类激活图（Eigen-CAM）进行决策可视化。实验结果表明，HGNet 达到了 94% 的准确率、90.6% 的召回率和 90% 的 mAP@0.5，显著提高了小病灶的区分能力和临床解释性。本文发表后，源代码将公开发布。 

---
# A Novel Approach for Estimating Positive Lyapunov Exponents in One-Dimensional Chaotic Time Series Using Machine Learning 

**Title (ZH)**: 一种基于机器学习的一维混沌时间序列正李雅普诺夫 exponent 估计的新方法 

**Authors**: A. Velichko, M. Belyaev, P. Boriskov  

**Link**: [PDF](https://arxiv.org/pdf/2507.04868)  

**Abstract**: Understanding and quantifying chaos in nonlinear dynamical systems remains a fundamental challenge in science and engineering. The Lyapunov exponent is a key measure of chaotic behavior, but its accurate estimation from experimental data is often hindered by methodological and computational limitations. In this work, we present a novel machine-learning-based approach for estimating the positive Lyapunov exponent (MLE) from one-dimensional time series, using the growth of out-of-sample prediction errors as a proxy for trajectory divergence. Our method demonstrates high scientific relevance, offering a robust, data-driven alternative to traditional analytic techniques. Through comprehensive testing on several canonical chaotic maps - including the logistic, sine, cubic, and Chebyshev maps - we achieved a coefficient of determination R2pos > 0.9 between predicted and theoretical MLE values for time series as short as M = 200 points. The best accuracy was observed for the Chebyshev map (R2pos = 0.999). Notably, the proposed method maintains high computational efficiency and generalizes well across various machine learning algorithms. These results highlight the significance of our approach for practical chaos analysis in both synthetic and experimental settings, opening new possibilities for robust nonlinear dynamics assessment when only time series data are available. 

**Abstract (ZH)**: 理解与定量分析非线性动力系统中的混沌现象仍然是科学和工程中的基本挑战。Lyapunov指数是衡量混沌行为的关键指标，但其从实验数据中准确估计常常受到方法论和计算限制的阻碍。在本工作中，我们提出了一种基于机器学习的方法，用于估计一维时间序列的正Lyapunov指数（MLE），采用离样预测误差的增长作为轨迹发散的代理。该方法具有高度的科学意义，提供了一种稳健的数据驱动替代传统分析技术。通过对包括逻辑映射、正弦映射、三次映射和Chebyshev映射在内的几个经典混沌映射进行全面测试，我们实现了预测和理论MLE值之间的决定系数R2pos > 0.9，对于时间序列长度仅为M = 200点的情况也是如此。Chebyshev映射达到了最佳准确性（R2pos = 0.999）。值得注意的是，所提出的方法保持了高计算效率，并且在各种机器学习算法中表现良好。这些结果突显了我们在合成和实验条件下进行实际混沌分析的重要性，为仅使用时间序列数据评估非线性动力学提供新的可能性。 

---
# Towards Human-in-the-Loop Onset Detection: A Transfer Learning Approach for Maracatu 

**Title (ZH)**: 基于人类在环的起音检测：一种转移学习方法应用于马拉卡图音乐 

**Authors**: António Sá Pinto  

**Link**: [PDF](https://arxiv.org/pdf/2507.04858)  

**Abstract**: We explore transfer learning strategies for musical onset detection in the Afro-Brazilian Maracatu tradition, which features complex rhythmic patterns that challenge conventional models. We adapt two Temporal Convolutional Network architectures: one pre-trained for onset detection (intra-task) and another for beat tracking (inter-task). Using only 5-second annotated snippets per instrument, we fine-tune these models through layer-wise retraining strategies for five traditional percussion instruments. Our results demonstrate significant improvements over baseline performance, with F1 scores reaching up to 0.998 in the intra-task setting and improvements of over 50 percentage points in best-case scenarios. The cross-task adaptation proves particularly effective for time-keeping instruments, where onsets naturally align with beat positions. The optimal fine-tuning configuration varies by instrument, highlighting the importance of instrument-specific adaptation strategies. This approach addresses the challenges of underrepresented musical traditions, offering an efficient human-in-the-loop methodology that minimizes annotation effort while maximizing performance. Our findings contribute to more inclusive music information retrieval tools applicable beyond Western musical contexts. 

**Abstract (ZH)**: 我们探讨了在巴伊亚马拉卡图传统音乐中迁移学习方法在节奏模式复杂的表现上的应用，该传统音乐挑战了传统的模型。我们调整了两种时间卷积网络架构：一种用于内部任务的预训练模型（用于起始点检测），另一种用于跨任务的节拍追踪。仅使用每件乐器5秒的标注片段，我们通过逐层重新训练策略对这些模型进行了微调，以适应五种传统打击乐器。我们的结果展示了显著的改进，在内部任务设置中F1分数达到了0.998，在最佳情况下性能提高了超过50个百分点。跨任务适应对于保持时间的乐器尤为有效，在这些乐器中起始点自然与节拍位置对齐。最佳微调配置因乐器而异，突显了乐器特定适应策略的重要性。这种方法解决了代表性不足的音乐传统所面临的挑战，提供了一种高效的人机交互方法，该方法最大限度地提高性能的同时减少标注工作量。我们的研究结果为更包容的音乐信息检索工具做出了贡献，这些工具不仅适用于西方音乐背景。 

---
# Fast-VGAN: Lightweight Voice Conversion with Explicit Control of F0 and Duration Parameters 

**Title (ZH)**: Fast-VGAN: 轻量级语音转换，具备明确的F0和时长参数控制 

**Authors**: Mathilde Abrassart, Nicolas Obin, Axel Roebel  

**Link**: [PDF](https://arxiv.org/pdf/2507.04817)  

**Abstract**: Precise control over speech characteristics, such as pitch, duration, and speech rate, remains a significant challenge in the field of voice conversion. The ability to manipulate parameters like pitch and syllable rate is an important element for effective identity conversion, but can also be used independently for voice transformation, achieving goals that were historically addressed by vocoder-based methods.
In this work, we explore a convolutional neural network-based approach that aims to provide means for modifying fundamental frequency (F0), phoneme sequences, intensity, and speaker identity. Rather than relying on disentanglement techniques, our model is explicitly conditioned on these factors to generate mel spectrograms, which are then converted into waveforms using a universal neural vocoder. Accordingly, during inference, F0 contours, phoneme sequences, and speaker embeddings can be freely adjusted, allowing for intuitively controlled voice transformations.
We evaluate our approach on speaker conversion and expressive speech tasks using both perceptual and objective metrics. The results suggest that the proposed method offers substantial flexibility, while maintaining high intelligibility and speaker similarity. 

**Abstract (ZH)**: 精确控制语音特性（如音调、持续时间和语速）在语音转换领域仍是一项重大挑战。调节音调和音节速率等参数的能力是有效身份转换的重要要素，但也可以独立用于语音转换，实现历史上由 vocoder 基方法解决的目标。

在本工作中，我们探索了一种基于卷积神经网络的方法，旨在提供修改基频（F0）、音素序列、强度和说话人身份的手段。我们的模型不依赖于分离技术，而是明确地根据这些因素生成 mel 频谱图，并使用通用神经 vocoder 转换为波形。因此，在推理过程中，F0 轮廓、音素序列和说话人嵌入可以自由调整，从而实现直观控制的语音转换。

我们在说话人转换和表现性语音任务上使用感知和客观指标评估了我们的方法。结果表明，所提出的方法具有很高的灵活性，同时保持了高可理解性和说话人相似性。 

---
# From Vision To Language through Graph of Events in Space and Time: An Explainable Self-supervised Approach 

**Title (ZH)**: 从时空事件图到视觉与语言：一种可解释的自监督方法 

**Authors**: Mihai Masala, Marius Leordeanu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04815)  

**Abstract**: The task of describing video content in natural language is commonly referred to as video captioning. Unlike conventional video captions, which are typically brief and widely available, long-form paragraph descriptions in natural language are scarce. This limitation of current datasets is due to the expensive human manual annotation required and to the highly challenging task of explaining the language formation process from the perspective of the underlying story, as a complex system of interconnected events in space and time. Through a thorough analysis of recently published methods and available datasets, we identify a general lack of published resources dedicated to the problem of describing videos in complex language, beyond the level of descriptions in the form of enumerations of simple captions. Furthermore, while state-of-the-art methods produce impressive results on the task of generating shorter captions from videos by direct end-to-end learning between the videos and text, the problem of explaining the relationship between vision and language is still beyond our reach. In this work, we propose a shared representation between vision and language, based on graphs of events in space and time, which can be obtained in an explainable and analytical way, to integrate and connect multiple vision tasks to produce the final natural language description. Moreover, we also demonstrate how our automated and explainable video description generation process can function as a fully automatic teacher to effectively train direct, end-to-end neural student pathways, within a self-supervised neuro-analytical system. We validate that our explainable neuro-analytical approach generates coherent, rich and relevant textual descriptions on videos collected from multiple varied datasets, using both standard evaluation metrics, human annotations and consensus from ensembles of state-of-the-art VLMs. 

**Abstract (ZH)**: 视频内容用自然语言描述的任务通常被称为视频字幕。不同于传统的视频字幕通常简短且容易获取，自然语言形式的长篇描述稀缺。这一当前数据集的限制源自于需要昂贵的人工手动标注和从复杂的时间和空间互联事件体系中解释语言形成过程这一高度挑战性的任务。通过对最近发表的方法和可用数据集的详细分析，我们发现缺乏专注于用复杂语言描述视频的公开资源，超过简单字幕描述的形式。此外，尽管现有最先进的方法可以通过视频和文本之间的直接端到端学习生成简短的字幕，但在从视觉到语言关系的解释方面，问题仍超出我们的能力范围。在本工作中，我们提出了一种基于时空事件图的视觉和语言共享表示，可以从可解释和分析的角度获取，以集成和连接多个视觉任务，生成最终的自然语言描述。此外，我们还展示了我们的自动化且可解释的视频描述生成过程可以作为完全自动的教师，在自监督神经分析系统中有效训练直接的端到端神经学生路径。我们验证了我们的可解释神经分析方法可以生成多个不同数据集收集的视频上的连贯、丰富且相关的文本描述，使用标准评估指标、人工注释和最先进的VLMs集成的共识进行验证。 

---
# A Survey of Pun Generation: Datasets, Evaluations and Methodologies 

**Title (ZH)**: 生成性惩罚研究综述：数据集、评估与方法 

**Authors**: Yuchen Su, Yonghua Zhu, Ruofan Wang, Zijian Huang, Diana Benavides-Prado, Michael Witbrock  

**Link**: [PDF](https://arxiv.org/pdf/2507.04793)  

**Abstract**: Pun generation seeks to creatively modify linguistic elements in text to produce humour or evoke double meanings. It also aims to preserve coherence and contextual appropriateness, making it useful in creative writing and entertainment across various media and contexts. Although pun generation has received considerable attention in computational linguistics, there is currently no dedicated survey that systematically reviews this specific area. To bridge this gap, this paper provides a comprehensive review of pun generation datasets and methods across different stages, including conventional approaches, deep learning techniques, and pre-trained language models. Additionally, we summarise both automated and human evaluation metrics used to assess the quality of pun generation. Finally, we discuss the research challenges and propose promising directions for future work. 

**Abstract (ZH)**: pun生成旨在创意性地修改文本中的语言元素以产生幽默或引发双关含义，同时力求保持连贯性和语境适宜性，使其在各种媒体和情境下的创意写作和娱乐中具有实用性。尽管pun生成在计算语言学领域受到了广泛关注，但目前尚无专门系统回顾这一特定领域的文献。为填补这一空白，本文提供了跨不同阶段的pun生成数据集和方法的全面回顾，包括传统方法、深度学习技术和预训练语言模型。此外，我们总结了评估pun生成质量的自动化和人类评估指标。最后，我们讨论了研究挑战，并提出了未来工作的有希望的方向。 

---
# Model Compression using Progressive Channel Pruning 

**Title (ZH)**: 渐进通道剪枝的模型压缩 

**Authors**: Jinyang Guo, Weichen Zhang, Wanli Ouyang, Dong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04792)  

**Abstract**: In this work, we propose a simple but effective channel pruning framework called Progressive Channel Pruning (PCP) to accelerate Convolutional Neural Networks (CNNs). In contrast to the existing channel pruning methods that prune channels only once per layer in a layer-by-layer fashion, our new progressive framework iteratively prunes a small number of channels from several selected layers, which consists of a three-step attempting-selecting-pruning pipeline in each iteration. In the attempting step, we attempt to prune a pre-defined number of channels from one layer by using any existing channel pruning methods and estimate the accuracy drop for this layer based on the labelled samples in the validation set. In the selecting step, based on the estimated accuracy drops for all layers, we propose a greedy strategy to automatically select a set of layers that will lead to less overall accuracy drop after pruning these layers. In the pruning step, we prune a small number of channels from these selected layers. We further extend our PCP framework to prune channels for the deep transfer learning methods like Domain Adversarial Neural Network (DANN), in which we effectively reduce the data distribution mismatch in the channel pruning process by using both labelled samples from the source domain and pseudo-labelled samples from the target domain. Our comprehensive experiments on two benchmark datasets demonstrate that our PCP framework outperforms the existing channel pruning approaches under both supervised learning and transfer learning settings. 

**Abstract (ZH)**: 一种渐进通道剪枝框架（PCP）以加速卷积神经网络 

---
# Interaction-Merged Motion Planning: Effectively Leveraging Diverse Motion Datasets for Robust Planning 

**Title (ZH)**: 交互融合运动规划：有效利用多样化运动数据集进行稳健规划 

**Authors**: Giwon Lee, Wooseong Jeong, Daehee Park, Jaewoo Jeong, Kuk-Jin Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2507.04790)  

**Abstract**: Motion planning is a crucial component of autonomous robot driving. While various trajectory datasets exist, effectively utilizing them for a target domain remains challenging due to differences in agent interactions and environmental characteristics. Conventional approaches, such as domain adaptation or ensemble learning, leverage multiple source datasets but suffer from domain imbalance, catastrophic forgetting, and high computational costs. To address these challenges, we propose Interaction-Merged Motion Planning (IMMP), a novel approach that leverages parameter checkpoints trained on different domains during adaptation to the target domain. IMMP follows a two-step process: pre-merging to capture agent behaviors and interactions, sufficiently extracting diverse information from the source domain, followed by merging to construct an adaptable model that efficiently transfers diverse interactions to the target domain. Our method is evaluated on various planning benchmarks and models, demonstrating superior performance compared to conventional approaches. 

**Abstract (ZH)**: 交互融合运动规划（IMMP）：一种适应目标领域的新型方法 

---
# From Imitation to Innovation: The Emergence of AI Unique Artistic Styles and the Challenge of Copyright Protection 

**Title (ZH)**: 从模仿到创新：AI独特艺术风格的 emergence 和版权保护挑战 

**Authors**: Zexi Jia, Chuanwei Huang, Yeshuang Zhu, Hongyan Fei, Ying Deng, Zhiqiang Yuan, Jiapei Zhang, Jinchao Zhang, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.04769)  

**Abstract**: Current legal frameworks consider AI-generated works eligible for copyright protection when they meet originality requirements and involve substantial human intellectual input. However, systematic legal standards and reliable evaluation methods for AI art copyrights are lacking. Through comprehensive analysis of legal precedents, we establish three essential criteria for determining distinctive artistic style: stylistic consistency, creative uniqueness, and expressive accuracy. To address these challenges, we introduce ArtBulb, an interpretable and quantifiable framework for AI art copyright judgment that combines a novel style description-based multimodal clustering method with multimodal large language models (MLLMs). We also present AICD, the first benchmark dataset for AI art copyright annotated by artists and legal experts. Experimental results demonstrate that ArtBulb outperforms existing models in both quantitative and qualitative evaluations. Our work aims to bridge the gap between the legal and technological communities and bring greater attention to the societal issue of AI art copyrights. 

**Abstract (ZH)**: 当前的法律框架认为，当AI生成的作品满足原创性要求并包含显著的人工智能输入时，可以获得版权保护。然而，系统的法律标准和可靠的AI艺术版权评估方法仍然缺乏。通过综合分析法律先例，我们确立了确定独特艺术风格的三个基本标准：风格一致性、创造性独特性和表达准确性。为应对这些挑战，我们引入了ArtBulb，这是一种可解释和可量化的方法，用于评估AI艺术版权，它结合了一种新颖的基于风格描述的多模态聚类方法和多模态大语言模型(MLLM)。我们还提出了由艺术家和法律专家标注的AICD，这是首个用于AI艺术版权评估的标准数据集。实验结果表明，ArtBulb在定量和定性评估中均优于现有模型。我们的工作旨在弥合法律和技术社区之间的差距，并引起社会各界对AI艺术版权问题的关注。 

---
# CoSteer: Collaborative Decoding-Time Personalization via Local Delta Steering 

**Title (ZH)**: CoSteer: 合作解码时个性化调整通过局部Delta调整 

**Authors**: Hang Lv, Sheng Liang, Hao Wang, Hongchao Gu, Yaxiong Wu, Wei Guo, Defu Lian, Yong Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.04756)  

**Abstract**: Personalized text generation has become crucial for adapting language models to diverse and evolving users' personal context across cultural, temporal, and contextual dimensions. While existing methods often rely on centralized fine-tuning or static preference alignment, they struggle to achieve real-time adaptation under resource constraints inherent to personal devices. This limitation creates a dilemma: large cloud-based models lack access to localized user-specific information, while small on-device models cannot match the generation quality of their cloud counterparts. To address this dichotomy, we present CoSteer, a novel collaborative framework that enables decoding-time personalization through localized delta steering. Our key insight lies in leveraging the logits difference between personal context-aware and -agnostic outputs from local small models as steering signals for cloud-based LLMs. Specifically, we formulate token-level optimization as an online learning problem, where local delta vectors dynamically adjust the remote LLM's logits within the on-device environment. This approach preserves privacy by transmitting only the final steered tokens rather than raw data or intermediate vectors, while maintaining cloud-based LLMs' general capabilities without fine-tuning. Through comprehensive experiments on various personalized generation tasks, we demonstrate that CoSteer effectively assists LLMs in generating personalized content by leveraging locally stored user profiles and histories, ensuring privacy preservation through on-device data processing while maintaining acceptable computational overhead. 

**Abstract (ZH)**: 个性化文本生成对于适应具有跨文化、时空和情境多样化个人背景的语言模型变得至关重要。现有方法往往依赖于集中式微调或静态偏好对齐，但在个人设备资源有限的情况下难以实现实时适应。这一限制导致了一个困境：大型基于云的模型缺乏本地化的用户特定信息访问，而小型本地设备模型也无法匹配其云 counterparts 的生成质量。为解决这一矛盾，我们提出了 CoSteer，一种新颖的协作框架，通过本地局部调整差分引导实现解码时的个性化。我们的核心洞察在于利用本地小型模型生成的带有和不带个人上下文感知输出的 logits 差异作为基于云的大规模语言模型的引导信号。具体而言，我们将标记级别优化形式化为一个在线学习问题，其中本地微调整向向量动态调整远程大规模语言模型的 logits，在设备环境中进行。此方法通过仅传输最终引导标记而不是原始数据或中间向量来保护隐私，同时维持基于云的大规模语言模型的一般能力而不进行微调。通过在各种个性化生成任务上的全面实验，我们证明了 CoSteer 通过利用本地存储的用户资料和历史记录有效辅助大规模语言模型生成个性化内容，并通过设备端数据处理保护隐私，同时保持可接受的计算开销。 

---
# Large Language Models for Network Intrusion Detection Systems: Foundations, Implementations, and Future Directions 

**Title (ZH)**: 大型语言模型在网络入侵检测系统中的应用：基础、实现与未来发展 

**Authors**: Shuo Yang, Xinran Zheng, Xinchen Zhang, Jinfeng Xu, Jinze Li, Donglin Xie, Weicai Long, Edith C.H. Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2507.04752)  

**Abstract**: Large Language Models (LLMs) have revolutionized various fields with their exceptional capabilities in understanding, processing, and generating human-like text. This paper investigates the potential of LLMs in advancing Network Intrusion Detection Systems (NIDS), analyzing current challenges, methodologies, and future opportunities. It begins by establishing a foundational understanding of NIDS and LLMs, exploring the enabling technologies that bridge the gap between intelligent and cognitive systems in AI-driven NIDS. While Intelligent NIDS leverage machine learning and deep learning to detect threats based on learned patterns, they often lack contextual awareness and explainability. In contrast, Cognitive NIDS integrate LLMs to process both structured and unstructured security data, enabling deeper contextual reasoning, explainable decision-making, and automated response for intrusion behaviors. Practical implementations are then detailed, highlighting LLMs as processors, detectors, and explainers within a comprehensive AI-driven NIDS pipeline. Furthermore, the concept of an LLM-centered Controller is proposed, emphasizing its potential to coordinate intrusion detection workflows, optimizing tool collaboration and system performance. Finally, this paper identifies critical challenges and opportunities, aiming to foster innovation in developing reliable, adaptive, and explainable NIDS. By presenting the transformative potential of LLMs, this paper seeks to inspire advancement in next-generation network security systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过其在理解和生成人类文本方面的出色能力，已经革新了各个领域。本文探讨了LLMs在推进网络入侵检测系统（NIDS）方面的发展潜力，分析了现有挑战、方法论和未来机遇。文章首先建立了对NIDS和LLMs的基本理解，探索了连接基于人工智能驱动的智能NIDS中智能系统和认知系统之间的使能技术。智能NIDS利用机器学习和深度学习根据学习到的模式检测威胁，但往往缺乏上下文意识和解释性。相比之下，认知NIDS整合了LLMs来处理结构化和非结构化的安全数据，从而实现更深入的上下文推理、可解释的决策和入侵行为的自动化响应。文章随后详细介绍了实用实施，突出了LLMs在综合的人工智能驱动NIDS管道中作为处理器、检测器和解释器的角色。此外，提出了以LLM为中心的控制器的概念，强调其在协调入侵检测工作流程、优化工具协作和系统性能方面的潜力。最后，本文指出了关键的挑战和机遇，旨在促进开发可靠、适应性强和可解释的NIDS的创新。通过展示LLMs的变革潜力，本文旨在激励下一代网络安全性系统的进步。 

---
# MCFormer: A Multi-Cost-Volume Network and Comprehensive Benchmark for Particle Image Velocimetry 

**Title (ZH)**: MCFormer: 多成本体积网络及颗粒图像 velocimetry 综合基准 

**Authors**: Zicheng Lin, Xiaoqiang Li, Yichao Wang, Chuan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04750)  

**Abstract**: Particle Image Velocimetry (PIV) is fundamental to fluid dynamics, yet deep learning applications face significant hurdles. A critical gap exists: the lack of comprehensive evaluation of how diverse optical flow models perform specifically on PIV data, largely due to limitations in available datasets and the absence of a standardized benchmark. This prevents fair comparison and hinders progress. To address this, our primary contribution is a novel, large-scale synthetic PIV benchmark dataset generated from diverse CFD simulations (JHTDB and Blasius). It features unprecedented variety in particle densities, flow velocities, and continuous motion, enabling, for the first time, a standardized and rigorous evaluation of various optical flow and PIV algorithms. Complementing this, we propose Multi Cost Volume PIV (MCFormer), a new deep network architecture leveraging multi-frame temporal information and multiple cost volumes, specifically designed for PIV's sparse nature. Our comprehensive benchmark evaluation, the first of its kind, reveals significant performance variations among adapted optical flow models and demonstrates that MCFormer significantly outperforms existing methods, achieving the lowest overall normalized endpoint error (NEPE). This work provides both a foundational benchmark resource essential for future PIV research and a state-of-the-art method tailored for PIV challenges. We make our benchmark dataset and code publicly available to foster future research in this area. 

**Abstract (ZH)**: 粒子图像 velocimetry (PIV) 是流体力学的基础，但深度学习应用面临重大挑战。一个关键的缺口在于缺乏对各种光学流模型在PIV数据上表现的全面评估，主要原因是可用数据集的局限性和缺乏标准化基准。这阻碍了公平比较和研究进步。为解决这一问题，我们的主要贡献是一个新的大规模合成PIV基准数据集，该数据集源自多样化的CFD模拟（JHTDB和Blasius），并具备前所未有的粒子密度、流速和连续运动的多样性，首次实现了对各种光学流和PIV算法的标准和严格的评估。此外，我们提出了Multi Cost Volume PIV (MCFormer)，一种利用多帧时空信息和多个成本体的新型深度网络架构，特别适合PIV的稀疏性质。我们的综合基准评估，是首次此类评估，揭示了适应的光学流模型之间显著的性能差异，并展示了MCFormer显著优于现有方法，实现了最低的整体归一化端点误差 (NEPE)。这项工作提供了未来PIV研究不可或缺的基础性基准资源和针对PIV挑战的最先进的方法。我们公开发布了基准数据集和代码，以促进该领域的未来研究。 

---
# Word stress in self-supervised speech models: A cross-linguistic comparison 

**Title (ZH)**: 自主监督语音模型中的重音：跨语言对比研究 

**Authors**: Martijn Bentum, Louis ten Bosch, Tomas O. Lentz  

**Link**: [PDF](https://arxiv.org/pdf/2507.04738)  

**Abstract**: In this paper we study word stress representations learned by self-supervised speech models (S3M), specifically the Wav2vec 2.0 model. We investigate the S3M representations of word stress for five different languages: Three languages with variable or lexical stress (Dutch, English and German) and two languages with fixed or demarcative stress (Hungarian and Polish). We train diagnostic stress classifiers on S3M embeddings and show that they can distinguish between stressed and unstressed syllables in read-aloud short sentences with high accuracy. We also tested language-specificity effects of S3M word stress. The results indicate that the word stress representations are language-specific, with a greater difference between the set of variable versus the set of fixed stressed languages. 

**Abstract (ZH)**: 本文研究了自监督语音模型（S3M）学习的单词重音表示，具体分析了Wav2vec 2.0模型。我们探讨了五种不同语言的S3M重音表示：三种具有可变或词汇重音的语言（荷兰语、英语和德语），以及两种具有固定或界标重音的语言（匈牙利语和波兰语）。我们在S3M嵌入上训练诊断重音分类器，并展示了它们能够以高精度区分朗读短句中的重读和未重读音节。我们还测试了S3M单词重音的特定语言效应。结果表明，单词重音表示具有语言特异性，可变重音语言与固定重音语言之间的差异更大。 

---
# Losing Control: Data Poisoning Attack on Guided Diffusion via ControlNet 

**Title (ZH)**: 失控：通过ControlNet对引导扩散的数据中毒攻击 

**Authors**: Raz Lapid, Almog Dubin  

**Link**: [PDF](https://arxiv.org/pdf/2507.04726)  

**Abstract**: Text-to-image diffusion models have achieved remarkable success in translating textual prompts into high-fidelity images. ControlNets further extend these models by allowing precise, image-based conditioning (e.g., edge maps, depth, pose), enabling fine-grained control over structure and style. However, their dependence on large, publicly scraped datasets -- and the increasing use of community-shared data for fine-tuning -- exposes them to stealthy data poisoning attacks. In this work, we introduce a novel data poisoning method that manipulates ControlNets to generate images containing specific content without any text triggers. By injecting poisoned samples -- each pairing a subtly triggered input with an NSFW target -- the model retains clean-prompt fidelity yet reliably produces NSFW outputs when the trigger is present. On large-scale, high-quality datasets, our backdoor achieves high attack success rate while remaining imperceptible in raw inputs. These results reveal a critical vulnerability in open-source ControlNets pipelines and underscore the need for robust data sanitization and defense mechanisms. 

**Abstract (ZH)**: 基于文本到图像扩散模型的隐蔽数据投毒攻击研究：操纵ControlNets生成带特定内容的图像而不包含任何文本触发词 

---
# Who's the Mole? Modeling and Detecting Intention-Hiding Malicious Agents in LLM-Based Multi-Agent Systems 

**Title (ZH)**: 谁是内鬼？基于LLM的多Agent系统中的意图隐藏恶意代理建模与检测 

**Authors**: Yizhe Xie, Congcong Zhu, Xinyue Zhang, Minghao Wang, Chi Liu, Minglu Zhu, Tianqing Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04724)  

**Abstract**: Multi-agent systems powered by Large Language Models (LLM-MAS) demonstrate remarkable capabilities in collaborative problem-solving. While LLM-MAS exhibit strong collaborative abilities, the security risks in their communication and coordination remain underexplored. We bridge this gap by systematically investigating intention-hiding threats in LLM-MAS, and design four representative attack paradigms that subtly disrupt task completion while maintaining high concealment. These attacks are evaluated in centralized, decentralized, and layered communication structures. Experiments conducted on six benchmark datasets, including MMLU, MMLU-Pro, HumanEval, GSM8K, arithmetic, and biographies, demonstrate that they exhibit strong disruptive capabilities. To identify these threats, we propose a psychology-based detection framework AgentXposed, which combines the HEXACO personality model with the Reid Technique, using progressive questionnaire inquiries and behavior-based monitoring. Experiments conducted on six types of attacks show that our detection framework effectively identifies all types of malicious behaviors. The detection rate for our intention-hiding attacks is slightly lower than that of the two baselines, Incorrect Fact Injection and Dark Traits Injection, demonstrating the effectiveness of intention concealment. Our findings reveal the structural and behavioral risks posed by intention-hiding attacks and offer valuable insights into securing LLM-based multi-agent systems through psychological perspectives, which contributes to a deeper understanding of multi-agent safety. The code and data are available at this https URL. 

**Abstract (ZH)**: 由大型语言模型驱动的多智能体系统（LLM-MAS）在协作问题解决方面展示出显著能力。尽管LLM-MAS展现出了强大的协作能力，但其通信和协调中的安全风险仍需进一步探索。我们通过系统地研究LLM-MAS中的意图隐藏威胁，设计了四种代表性的攻击范式，这些攻击在任务完成过程中微妙地造成干扰同时保持高度隐蔽性。这些攻击在集中式、去中心化和多层次通信结构中进行了评估。实验结果显示，这些攻击具有较强的破坏能力。为了识别这些威胁，我们提出了一种基于心理学的检测框架AgentXposed，该框架将HEXACO人格模型与Reid技术结合起来，通过渐进式问卷调查和基于行为的监控。实验结果显示，我们的检测框架有效识别了六种不同类型的恶意行为。对于意图隐藏攻击的检测率略低于两个 baselines（错误事实注入和黑暗特质注入）的检测率，这表明意图隐藏的有效性。我们的研究揭示了意图隐藏攻击带来的结构和行为风险，并从心理学角度提供了关于如何通过多智能体系统安全性的宝贵见解，增进了对多智能体安全性的理解。代码和数据可在以下链接获取。 

---
# Geometric-Guided Few-Shot Dental Landmark Detection with Human-Centric Foundation Model 

**Title (ZH)**: 基于几何引导的以人类为中心的 Few-Shot 牙科标志点检测 

**Authors**: Anbang Wang, Marawan Elbatel, Keyuan Liu, Lizhuo Lin, Meng Lan, Yanqi Yang, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.04710)  

**Abstract**: Accurate detection of anatomic landmarks is essential for assessing alveolar bone and root conditions, thereby optimizing clinical outcomes in orthodontics, periodontics, and implant dentistry. Manual annotation of landmarks on cone-beam computed tomography (CBCT) by dentists is time-consuming, labor-intensive, and subject to inter-observer variability. Deep learning-based automated methods present a promising approach to streamline this process efficiently. However, the scarcity of training data and the high cost of expert annotations hinder the adoption of conventional deep learning techniques. To overcome these challenges, we introduce GeoSapiens, a novel few-shot learning framework designed for robust dental landmark detection using limited annotated CBCT of anterior teeth. Our GeoSapiens framework comprises two key components: (1) a robust baseline adapted from Sapiens, a foundational model that has achieved state-of-the-art performance in human-centric vision tasks, and (2) a novel geometric loss function that improves the model's capacity to capture critical geometric relationships among anatomical structures. Experiments conducted on our collected dataset of anterior teeth landmarks revealed that GeoSapiens surpassed existing landmark detection methods, outperforming the leading approach by an 8.18% higher success detection rate at a strict 0.5 mm threshold-a standard widely recognized in dental diagnostics. Code is available at: this https URL. 

**Abstract (ZH)**: 准确检测解剖标志对于评估牙槽骨和牙根状况至关重要，从而优化正畸、牙周病学和种植牙临床效果。牙医在锥形束计算机断层扫描（CBCT）上手动标注解剖标志耗时、费力且存在观察者间变异。基于深度学习的自动方法提供了一种有效简化这一过程的有前途的方法。然而，训练数据稀缺和专家标注的高成本阻碍了传统深度学习技术的广泛应用。为克服这些挑战，我们提出了一种名为GeoSapiens的新颖少量学习框架，用于使用有限注释的前牙CBCT进行 robust 牙齿解剖标志检测。GeoSapiens框架包括两个关键组件：(1) 从在人类中心视觉任务中表现优异的基础模型Sapiens改编而来的稳健基线，以及(2) 一种新颖的几何损失函数，该函数提高了模型捕捉解剖结构之间关键几何关系的能力。在我们收集的前牙解剖标志数据集上进行的实验表明，GeoSapiens超越了现有检测方法，在严格的0.5 mm阈值下成功检测率高出8.18%，该标准在牙科诊断中广受认可。代码可用于此：this https URL。 

---
# UrbanMind: Towards Urban General Intelligence via Tool-Enhanced Retrieval-Augmented Generation and Multilevel Optimization 

**Title (ZH)**: UrbanMind：通过工具增强的检索增强生成和多层优化 toward 城市通用智能 

**Authors**: Kai Yang, Zelin Zhu, Chengtao Jian, Hui Ma, Shengjie Zhao, Xiaozhou Ye, Ye Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04706)  

**Abstract**: Urban general intelligence (UGI) refers to the capacity of AI systems to autonomously perceive, reason, and act within dynamic and complex urban environments. In this paper, we introduce UrbanMind, a tool-enhanced retrieval-augmented generation (RAG) framework designed to facilitate UGI. Central to UrbanMind is a novel architecture based on Continual Retrieval-Augmented MoE-based LLM (C-RAG-LLM), which dynamically incorporates domain-specific knowledge and evolving urban data to support long-term adaptability. The architecture of C-RAG-LLM aligns naturally with a multilevel optimization framework, where different layers are treated as interdependent sub-problems. Each layer has distinct objectives and can be optimized either independently or jointly through a hierarchical learning process. The framework is highly flexible, supporting both end-to-end training and partial layer-wise optimization based on resource or deployment constraints. To remain adaptive under data drift, it is further integrated with an incremental corpus updating mechanism. Evaluations on real-world urban tasks of a variety of complexity verify the effectiveness of the proposed framework. This work presents a promising step toward the realization of general-purpose LLM agents in future urban environments. 

**Abstract (ZH)**: 城市通用智能（UGI）是指AI系统在动态复杂的城市环境中自主感知、推理和行动的能力。本文介绍了UrbanMind，一种工具增强的检索增强生成（RAG）框架，旨在促进UGI的实现。UrbanMind的核心是一种新颖的基于连续检索增强MoE基大语言模型（C-RAG-LLM）的架构，该架构能够动态地纳入领域特定知识和不断变化的城市数据，以支持长期适应性。C-RAG-LLM架构自然地与多级优化框架相契合，其中不同层次被视为相互依赖的子问题。每一层都有其独特的目标，并可以通过分层学习过程独立或联合进行优化。该框架具有高度灵活性，可以根据资源或部署约束支持端到端训练和部分层次优化。为保持在数据漂移下的适用性，进一步集成了增量语料库更新机制。对复杂程度各异的现实城市任务的评估验证了所提出框架的有效性。本文展示了向未来城市环境中实现通用大语言模型代理的重要一步。 

---
# SPATIA: Multimodal Model for Prediction and Generation of Spatial Cell Phenotypes 

**Title (ZH)**: SPATIA：多模态模型用于空间细胞表型的预测与生成 

**Authors**: Zhenglun Kong, Mufan Qiu, John Boesen, Xiang Lin, Sukwon Yun, Tianlong Chen, Manolis Kellis, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2507.04704)  

**Abstract**: Understanding how cellular morphology, gene expression, and spatial organization jointly shape tissue function is a central challenge in biology. Image-based spatial transcriptomics technologies now provide high-resolution measurements of cell images and gene expression profiles, but machine learning methods typically analyze these modalities in isolation or at limited resolution. We address the problem of learning unified, spatially aware representations that integrate cell morphology, gene expression, and spatial context across biological scales. This requires models that can operate at single-cell resolution, reason across spatial neighborhoods, and generalize to whole-slide tissue organization. Here, we introduce SPATIA, a multi-scale generative and predictive model for spatial transcriptomics. SPATIA learns cell-level embeddings by fusing image-derived morphological tokens and transcriptomic vector tokens using cross-attention and then aggregates them at niche and tissue levels using transformer modules to capture spatial dependencies. SPATIA incorporates token merging in its generative diffusion decoder to synthesize high-resolution cell images conditioned on gene expression. We assembled a multi-scale dataset consisting of 17 million cell-gene pairs, 1 million niche-gene pairs, and 10,000 tissue-gene pairs across 49 donors, 17 tissue types, and 12 disease states. We benchmark SPATIA against 13 existing models across 12 individual tasks, which span several categories including cell annotation, cell clustering, gene imputation, cross-modal prediction, and image generation. SPATIA achieves improved performance over all baselines and generates realistic cell morphologies that reflect transcriptomic perturbations. 

**Abstract (ZH)**: 基于图像的空间转录组学多尺度生成和预测模型SPATIA 

---
# Tempo-R0: A Video-MLLM for Temporal Video Grounding through Efficient Temporal Sensing Reinforcement Learning 

**Title (ZH)**: Tempo-R0：一种通过高效时间感知强化学习进行-temporal视频约束的视频-MLLM 

**Authors**: Feng Yue, Zhaoxing Zhang, Junming Jiao, Zhengyu Liang, Shiwen Cao, Feifei Zhang, Rong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.04702)  

**Abstract**: Temporal Video Grounding (TVG), which requires pinpointing relevant temporal segments from video based on language query, has always been a highly challenging task in the field of video understanding. Videos often have a larger volume of information and redundancy than texts or images. Models should present comprehensive understanding of the whole video to accurately retrieve query-relevant clips. We thus propose Tempo-R0: a Video Multimodal Large Language Model (Video-MLLM) for the temporal video grounding task via multimodal temporal sensing reinforcement. Specifically, during the preprocessing stage of our pipeline, we employ Self-adaptive Attention Allocation (SAA) method based on frame content variation to efficiently use the MLLM's limited attention. The Explicit Timestamp-modal Aligned (ETA) method is also utilized to strengthen our model's capability to perceive the boundaries of events in the video. In the fine-tuning part of our pipeline, we creatively apply Partial Irrelevance Refusing-based Group Relative Policy Optimization (PIR-GRPO) in TVG area to foster model's temporal reasoning from not only accepting relevant video-query pairs but also refusing irrelevant ones. Experiments demonstrate that our method accomplishes a notable advantage over SOTA solutions by around 3.5% on both the original QVHighlights testbench and its corrected version with more reasonable ground truth annotations. 

**Abstract (ZH)**: 基于语言查询的视频时序定位（Temporal Video Grounding, TVG）：多模态时空感知强化的视频大规模语言模型（Video-MLLM） 

---
# Bridging KAN and MLP: MJKAN, a Hybrid Architecture with Both Efficiency and Expressiveness 

**Title (ZH)**: 将KAN和MLP桥梁化：MJKAN，一种兼具效率与表达性的混合架构 

**Authors**: Hanseon Joo, Hayoung Choi, Ook Lee, Minjong Cheon  

**Link**: [PDF](https://arxiv.org/pdf/2507.04690)  

**Abstract**: Kolmogorov-Arnold Networks (KANs) have garnered attention for replacing fixed activation functions with learnable univariate functions, but they exhibit practical limitations, including high computational costs and performance deficits in general classification tasks. In this paper, we propose the Modulation Joint KAN (MJKAN), a novel neural network layer designed to overcome these challenges. MJKAN integrates a FiLM (Feature-wise Linear Modulation)-like mechanism with Radial Basis Function (RBF) activations, creating a hybrid architecture that combines the non-linear expressive power of KANs with the efficiency of Multilayer Perceptrons (MLPs). We empirically validated MJKAN's performance across a diverse set of benchmarks, including function regression, image classification (MNIST, CIFAR-10/100), and natural language processing (AG News, SMS Spam). The results demonstrate that MJKAN achieves superior approximation capabilities in function regression tasks, significantly outperforming MLPs, with performance improving as the number of basis functions increases. Conversely, in image and text classification, its performance was competitive with MLPs but revealed a critical dependency on the number of basis functions. We found that a smaller basis size was crucial for better generalization, highlighting that the model's capacity must be carefully tuned to the complexity of the data to prevent overfitting. In conclusion, MJKAN offers a flexible architecture that inherits the theoretical advantages of KANs while improving computational efficiency and practical viability. 

**Abstract (ZH)**: Modulation Joint Kolmogorov-Arnold Networks 

---
# Identify, Isolate, and Purge: Mitigating Hallucinations in LVLMs via Self-Evolving Distillation 

**Title (ZH)**: 识别、隔离和净化：通过自我进化的蒸馏技术减轻LVLM中的幻觉 

**Authors**: Wenhao Li, Xiu Su, Jingyi Wu, Feng Yang, Yang Liu, Yi Chen, Shan You, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04680)  

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable advancements in numerous areas such as multimedia. However, hallucination issues significantly limit their credibility and application potential. Existing mitigation methods typically rely on external tools or the comparison of multi-round inference, which significantly increase inference time. In this paper, we propose \textbf{SE}lf-\textbf{E}volving \textbf{D}istillation (\textbf{SEED}), which identifies hallucinations within the inner knowledge of LVLMs, isolates and purges them, and then distills the purified knowledge back into the model, enabling self-evolution. Furthermore, we identified that traditional distillation methods are prone to inducing void spaces in the output space of LVLMs. To address this issue, we propose a Mode-Seeking Evolving approach, which performs distillation to capture the dominant modes of the purified knowledge distribution, thereby avoiding the chaotic results that could emerge from void spaces. Moreover, we introduce a Hallucination Elimination Adapter, which corrects the dark knowledge of the original model by learning purified knowledge. Extensive experiments on multiple benchmarks validate the superiority of our SEED, demonstrating substantial improvements in mitigating hallucinations for representative LVLM models such as LLaVA-1.5 and InternVL2. Remarkably, the F1 score of LLaVA-1.5 on the hallucination evaluation metric POPE-Random improved from 81.3 to 88.3. 

**Abstract (ZH)**: SEED: Self-Evolving Distillation for Hallucination Mitigation in Large Vision-Language Models 

---
# What's Making That Sound Right Now? Video-centric Audio-Visual Localization 

**Title (ZH)**: 当前是什么声音？基于视频的音视频定位 

**Authors**: Hahyeon Choi, Junhoo Lee, Nojun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2507.04667)  

**Abstract**: Audio-Visual Localization (AVL) aims to identify sound-emitting sources within a visual scene. However, existing studies focus on image-level audio-visual associations, failing to capture temporal dynamics. Moreover, they assume simplified scenarios where sound sources are always visible and involve only a single object. To address these limitations, we propose AVATAR, a video-centric AVL benchmark that incorporates high-resolution temporal information. AVATAR introduces four distinct scenarios -- Single-sound, Mixed-sound, Multi-entity, and Off-screen -- enabling a more comprehensive evaluation of AVL models. Additionally, we present TAVLO, a novel video-centric AVL model that explicitly integrates temporal information. Experimental results show that conventional methods struggle to track temporal variations due to their reliance on global audio features and frame-level mappings. In contrast, TAVLO achieves robust and precise audio-visual alignment by leveraging high-resolution temporal modeling. Our work empirically demonstrates the importance of temporal dynamics in AVL and establishes a new standard for video-centric audio-visual localization. 

**Abstract (ZH)**: 音频-视觉定位（AVL）旨在识别视觉场景中的声源。然而，现有研究关注图像级别的音频-视觉关联，未能捕捉时间动态性。此外，它们假设声源始终可见且仅涉及单一物体的简化场景。为解决这些限制，我们提出AVATAR，一个以视频为中心的AVL基准，包含高分辨率时间信息。AVATAR引入了四种不同的场景——单声源、混合声源、多实体和离屏——以实现更全面的AVL模型评估。此外，我们还提出了TAVLO，一种以视频为中心的AVL模型，明确整合时间信息。实验结果表明，传统方法由于依赖全局音频特征和帧级映射，难以跟踪时间变化。相比之下，TAVLO通过利用高分辨率的时间建模实现稳健且精确的音频-视觉对齐。我们的工作从经验上证明了时间动态性在AVL中的重要性，并建立了以视频为中心的音频-视觉定位的新标准。 

---
# LTMSformer: A Local Trend-Aware Attention and Motion State Encoding Transformer for Multi-Agent Trajectory Prediction 

**Title (ZH)**: LTMSformer：一种考虑局部趋势关注和运动状态编码的多agents轨迹预测Transformer模型 

**Authors**: Yixin Yan, Yang Li, Yuanfan Wang, Xiaozhou Zhou, Beihao Xia, Manjiang Hu, Hongmao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2507.04634)  

**Abstract**: It has been challenging to model the complex temporal-spatial dependencies between agents for trajectory prediction. As each state of an agent is closely related to the states of adjacent time steps, capturing the local temporal dependency is beneficial for prediction, while most studies often overlook it. Besides, learning the high-order motion state attributes is expected to enhance spatial interaction modeling, but it is rarely seen in previous works. To address this, we propose a lightweight framework, LTMSformer, to extract temporal-spatial interaction features for multi-modal trajectory prediction. Specifically, we introduce a Local Trend-Aware Attention mechanism to capture the local temporal dependency by leveraging a convolutional attention mechanism with hierarchical local time boxes. Next, to model the spatial interaction dependency, we build a Motion State Encoder to incorporate high-order motion state attributes, such as acceleration, jerk, heading, etc. To further refine the trajectory prediction, we propose a Lightweight Proposal Refinement Module that leverages Multi-Layer Perceptrons for trajectory embedding and generates the refined trajectories with fewer model parameters. Experiment results on the Argoverse 1 dataset demonstrate that our method outperforms the baseline HiVT-64, reducing the minADE by approximately 4.35%, the minFDE by 8.74%, and the MR by 20%. We also achieve higher accuracy than HiVT-128 with a 68% reduction in model size. 

**Abstract (ZH)**: 基于时空交互特征的轻量级框架LTMSformer及其在多模态轨迹预测中的应用 

---
# Learning Robust Stereo Matching in the Wild with Selective Mixture-of-Experts 

**Title (ZH)**: 在野外学习具有选择性混合专家的稳健立体匹配 

**Authors**: Yun Wang, Longguang Wang, Chenghao Zhang, Yongjian Zhang, Zhanjie Zhang, Ao Ma, Chenyou Fan, Tin Lun Lam, Junjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04631)  

**Abstract**: Recently, learning-based stereo matching networks have advanced significantly. However, they often lack robustness and struggle to achieve impressive cross-domain performance due to domain shifts and imbalanced disparity distributions among diverse datasets. Leveraging Vision Foundation Models (VFMs) can intuitively enhance the model's robustness, but integrating such a model into stereo matching cost-effectively to fully realize their robustness remains a key challenge. To address this, we propose SMoEStereo, a novel framework that adapts VFMs for stereo matching through a tailored, scene-specific fusion of Low-Rank Adaptation (LoRA) and Mixture-of-Experts (MoE) modules. SMoEStereo introduces MoE-LoRA with adaptive ranks and MoE-Adapter with adaptive kernel sizes. The former dynamically selects optimal experts within MoE to adapt varying scenes across domains, while the latter injects inductive bias into frozen VFMs to improve geometric feature extraction. Importantly, to mitigate computational overhead, we further propose a lightweight decision network that selectively activates MoE modules based on input complexity, balancing efficiency with accuracy. Extensive experiments demonstrate that our method exhibits state-of-the-art cross-domain and joint generalization across multiple benchmarks without dataset-specific adaptation. The code is available at \textcolor{red}{this https URL}. 

**Abstract (ZH)**: 基于视觉基础模型的 stereo 匹配新框架：SMoEStereo 

---
# Knowledge-Aware Self-Correction in Language Models via Structured Memory Graphs 

**Title (ZH)**: 基于结构记忆图的语言模型知识感知自修正 

**Authors**: Swayamjit Saha  

**Link**: [PDF](https://arxiv.org/pdf/2507.04625)  

**Abstract**: Large Language Models (LLMs) are powerful yet prone to generating factual errors, commonly referred to as hallucinations. We present a lightweight, interpretable framework for knowledge-aware self-correction of LLM outputs using structured memory graphs based on RDF triples. Without retraining or fine-tuning, our method post-processes model outputs and corrects factual inconsistencies via external semantic memory. We demonstrate the approach using DistilGPT-2 and show promising results on simple factual prompts. 

**Abstract (ZH)**: 大型语言模型（LLMs）具有强大的能力，但却容易生成事实错误，通常称为幻觉。我们提出了一个基于RDF三元组的结构化记忆图的轻量级可解释框架，用于知识导向的LLM输出自矫正。不需重新训练或微调，该方法对模型输出进行后处理，并通过外部语义记忆矫正事实不一致。我们使用DistilGPT-2进行了演示，并在简单事实提示上取得了令人鼓舞的结果。 

---
# Hierarchical Intent-guided Optimization with Pluggable LLM-Driven Semantics for Session-based Recommendation 

**Title (ZH)**: 基于层次化意图引导的可插拔LLM驱动语义会话推荐优化 

**Authors**: Jinpeng Chen, Jianxiang He, Huan Li, Senzhang Wang, Yuan Cao, Kaimin Wei, Zhenye Yang, Ye Ji  

**Link**: [PDF](https://arxiv.org/pdf/2507.04623)  

**Abstract**: Session-based Recommendation (SBR) aims to predict the next item a user will likely engage with, using their interaction sequence within an anonymous session. Existing SBR models often focus only on single-session information, ignoring inter-session relationships and valuable cross-session insights. Some methods try to include inter-session data but struggle with noise and irrelevant information, reducing performance. Additionally, most models rely on item ID co-occurrence and overlook rich semantic details, limiting their ability to capture fine-grained item features. To address these challenges, we propose a novel hierarchical intent-guided optimization approach with pluggable LLM-driven semantic learning for session-based recommendations, called HIPHOP. First, we introduce a pluggable embedding module based on large language models (LLMs) to generate high-quality semantic representations, enhancing item embeddings. Second, HIPHOP utilizes graph neural networks (GNNs) to model item transition relationships and incorporates a dynamic multi-intent capturing module to address users' diverse interests within a session. Additionally, we design a hierarchical inter-session similarity learning module, guided by user intent, to capture global and local session relationships, effectively exploring users' long-term and short-term interests. To mitigate noise, an intent-guided denoising strategy is applied during inter-session learning. Finally, we enhance the model's discriminative capability by using contrastive learning to optimize session representations. Experiments on multiple datasets show that HIPHOP significantly outperforms existing methods, demonstrating its effectiveness in improving recommendation quality. Our code is available: this https URL. 

**Abstract (ZH)**: 基于会话的推荐（SBR）旨在使用用户在匿名会话内的交互序列来预测用户可能 engagement 的下一个项目。现有的SBR模型通常仅关注单会话信息，忽视了会话间的关系和跨会话有价值的见解。一些方法尝试包含会话间数据，但难以处理噪音和无关信息，影响了性能。此外，大多数模型依赖于物品ID共现，忽略了丰富的语义细节，限制了它们捕捉细粒度物品特征的能力。为了解决这些挑战，我们提出了一种名为HIPHOP的新颖层次意图引导优化方法，该方法结合可插拔的大语言模型驱动的语义学习，用于基于会话的推荐。首先，我们基于大规模语言模型引入了一个可插拔的嵌入模块，生成高质量的语义表示，增强物品嵌入。其次，HIPHOP利用图神经网络（GNNs）建模项目转换关系，并结合一个动态多意图捕捉模块以应对会话内用户 varied 的兴趣。此外，我们设计了一个层次化的会话间相似性学习模块，受用户意图引导，以捕捉全局和局部会话关系，有效探索用户长期和短期兴趣。为了减轻噪音，在会话间学习期间应用意图引导的去噪策略。最后，通过对比学习增强模型的辨别能力，优化会话表示。在多个数据集上的实验结果显示，HIPHOP显著优于现有方法，证明了其在提高推荐质量方面的有效性。我们的代码可在以下链接获取：this https URL。 

---
# Multimodal LLM Integrated Semantic Communications for 6G Immersive Experiences 

**Title (ZH)**: 多模态LLM集成语义通信以实现6G沉浸式体验 

**Authors**: Yusong Zhang, Yuxuan Sun, Lei Guo, Wei Chen, Bo Ai, Deniz Gunduz  

**Link**: [PDF](https://arxiv.org/pdf/2507.04621)  

**Abstract**: 6G networks promise revolutionary immersive communication experiences including augmented reality (AR), virtual reality (VR), and holographic communications. These applications demand high-dimensional multimodal data transmission and intelligent data processing in real-time, which is extremely challenging over resource-limited wireless communication systems. Moreover, a joint understanding of the environment, context, and user intent is essential to deliver task-relevant content effectively. This article presents a novel multimodal large language model (MLLM) integrated semantic communications framework, termed MLLM-SC, which fully leverages reasoning and generative capabilities of pre-trained foundation models for context-aware and task-oriented wireless communication. The MLLM-SC framework adopts a device-edge collaborative architecture. At the edge, MLLM-empowered semantic guidance module analyzes multimodal inputs, user intents, and channel conditions to generate importance-aware attention maps prioritizing semantically critical information. An importance-aware semantic encoder and a resource-adaptive semantic decoder are jointly designed and optimized, which can utilize the semantic guidance for adaptive bandwidth allocation and high-quality content reconstruction or generation. Extensive case studies on visual question answering for AR/VR applications and diffusion-driven image generation validate the effectiveness of MLLM-SC. 

**Abstract (ZH)**: 6G网络承诺提供革命性的沉浸式通信体验，包括增强现实（AR）、虚拟现实（VR）和全息通信。这些应用要求在资源有限的无线通信系统中进行实时的高维多模态数据传输和智能数据处理，这极具挑战性。此外，理解和联合环境、上下文以及用户意图对于有效提供任务相关的内容至关重要。本文提出了一种新颖的多模态大型语言模型（MLLM）集成语义通信框架，称为MLLM-SC，该框架充分利用预训练基础模型的推理和生成能力，实现具有上下文感知和任务导向的无线通信。MLLM-SC框架采用设备-边缘协作架构。在边缘处，MLLM赋能的语义指导模块分析多模态输入、用户意图和信道条件，生成注重语义关键信息的重要性感知注意力图。重要性感知语义编码器和资源自适应语义解码器被联合设计和优化，能够利用语义指导进行自适应带宽分配和高质量内容的重建或生成。广泛应用于AR/VR应用的视觉问答和扩散驱动图像生成的案例研究验证了MLLM-SC的有效性。 

---
# Information-Guided Diffusion Sampling for Dataset Distillation 

**Title (ZH)**: 信息引导的扩散采样用于数据集提炼 

**Authors**: Linfeng Ye, Shayan Mohajer Hamidi, Guang Li, Takahiro Ogawa, Miki Haseyama, Konstantinos N. Plataniotis  

**Link**: [PDF](https://arxiv.org/pdf/2507.04619)  

**Abstract**: Dataset distillation aims to create a compact dataset that retains essential information while maintaining model performance. Diffusion models (DMs) have shown promise for this task but struggle in low images-per-class (IPC) settings, where generated samples lack diversity. In this paper, we address this issue from an information-theoretic perspective by identifying two key types of information that a distilled dataset must preserve: ($i$) prototype information $\mathrm{I}(X;Y)$, which captures label-relevant features; and ($ii$) contextual information $\mathrm{H}(X | Y)$, which preserves intra-class variability. Here, $(X,Y)$ represents the pair of random variables corresponding to the input data and its ground truth label, respectively. Observing that the required contextual information scales with IPC, we propose maximizing $\mathrm{I}(X;Y) + \beta \mathrm{H}(X | Y)$ during the DM sampling process, where $\beta$ is IPC-dependent. Since directly computing $\mathrm{I}(X;Y)$ and $\mathrm{H}(X | Y)$ is intractable, we develop variational estimations to tightly lower-bound these quantities via a data-driven approach. Our approach, information-guided diffusion sampling (IGDS), seamlessly integrates with diffusion models and improves dataset distillation across all IPC settings. Experiments on Tiny ImageNet and ImageNet subsets show that IGDS significantly outperforms existing methods, particularly in low-IPC regimes. The code will be released upon acceptance. 

**Abstract (ZH)**: 基于信息论的Dataset蒸馏：最大化关键信息保留的扩散模型采样方法 

---
# HiLa: Hierarchical Vision-Language Collaboration for Cancer Survival Prediction 

**Title (ZH)**: HiLa: 分层视觉-语言协作预测癌症生存率 

**Authors**: Jiaqi Cui, Lu Wen, Yuchen Fei, Bo Liu, Luping Zhou, Dinggang Shen, Yan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04613)  

**Abstract**: Survival prediction using whole-slide images (WSIs) is crucial in cancer re-search. Despite notable success, existing approaches are limited by their reliance on sparse slide-level labels, which hinders the learning of discriminative repre-sentations from gigapixel WSIs. Recently, vision language (VL) models, which incorporate additional language supervision, have emerged as a promising solu-tion. However, VL-based survival prediction remains largely unexplored due to two key challenges. First, current methods often rely on only one simple lan-guage prompt and basic cosine similarity, which fails to learn fine-grained associ-ations between multi-faceted linguistic information and visual features within WSI, resulting in inadequate vision-language alignment. Second, these methods primarily exploit patch-level information, overlooking the intrinsic hierarchy of WSIs and their interactions, causing ineffective modeling of hierarchical interac-tions. To tackle these problems, we propose a novel Hierarchical vision-Language collaboration (HiLa) framework for improved survival prediction. Specifically, HiLa employs pretrained feature extractors to generate hierarchical visual features from WSIs at both patch and region levels. At each level, a series of language prompts describing various survival-related attributes are constructed and aligned with visual features via Optimal Prompt Learning (OPL). This ap-proach enables the comprehensive learning of discriminative visual features cor-responding to different survival-related attributes from prompts, thereby improv-ing vision-language alignment. Furthermore, we introduce two modules, i.e., Cross-Level Propagation (CLP) and Mutual Contrastive Learning (MCL) to maximize hierarchical cooperation by promoting interactions and consistency be-tween patch and region levels. Experiments on three TCGA datasets demonstrate our SOTA performance. 

**Abstract (ZH)**: 使用玻片图像进行生存预测对于癌症研究至关重要。尽管现有方法取得了显著成果，但它们依赖于稀疏的切片级别标签，这限制了从 gigapixel 玻片图像中学习判别性表示的能力。最近，结合了额外语言监督的视觉语言（VL）模型 emerged 作为有希望的解决方案。然而，基于 VL 的生存预测尚未充分探索，主要因为两个关键挑战：首先，当前方法通常仅依赖单一简单的语言提示和基础的余弦相似度，无法学习多面语言信息与玻片图像视觉特征之间的精细关联，导致不充分的视觉-语言对齐；其次，这些方法主要利用 patch 级信息，忽略了玻片图像固有的层级结构及其交互，导致层级交互建模效果不佳。为解决这些问题，我们提出了一个新的层次视觉-语言协作（HiLa）框架以改进生存预测。具体而言，HiLa 使用预训练的特征提取器从玻片图像的 patch 和区域级别生成层次视觉特征。在每一级，一系列描述各种生存相关属性的语言提示被构建，并通过 Optimal Prompt Learning（OPL）与视觉特征对齐。这种方法使不同生存相关属性对应的判别性视觉特征的全面学习成为可能，从而改善视觉-语言对齐。此外，我们引入了两个模块，即跨级传播（CLP）和互式对比学习（MCL），以通过促进 patch 和区域级别的交互与一致性，最大化层级协作。在三个 TCGA 数据集上的实验表明了我们的 SOTA 性能。 

---
# any4: Learned 4-bit Numeric Representation for LLMs 

**Title (ZH)**: 任何4：LLMs的learned 4-bit 数值表示 

**Authors**: Mostafa Elhoushi, Jeff Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2507.04610)  

**Abstract**: We present any4, a learned 4-bit weight quantization solution for large language models (LLMs) providing arbitrary numeric representations without requiring pre-processing of weights or activations. any4 yields higher accuracy compared to other related 4-bit numeric representation types: int4, fp4 and nf4, as evaluated on a range of model sizes, generations and families (Llama 2, Llama 3, Mistral and Mixtral). While any4 does not require preprocessing of weights or activations, it is also competitive with orthogonal techniques that require such preprocessing (e.g., AWQ and GPTQ). We also experiment with any3 and any2 and show competitiveness at lower bits. Additionally, we show that we can calibrate using a single curated diverse sample rather than hundreds of samples from a dataset as done in most quantization approaches. We also open source tinygemm, a latency optimized GPU matrix multiplication library for LLMs, that implements any4 using a GPU-efficient lookup table strategy along with other common quantization methods. We open source our code at this https URL . 

**Abstract (ZH)**: 我们介绍了any4，这是一种用于大型语言模型的4位权重量化解决方案，无需预处理权重或激活值即可提供任意数值表示，并且在多种模型规模、生成和家族（Llama 2、Llama 3、Mistral 和 Mixtral）的评估中，其准确度高于其他相关4位数值表示类型（int4、fp4 和 nf4）。此外，我们还试验了any3和any2，并展示了在较低位数下的竞争力。我们还展示了使用单一精心选择的多样样本进行校准的方法，而大多数量化方法则需要使用数据集中的数百个样本。我们还开源了针对大型语言模型优化延迟的tinygemm GPU矩阵乘法库，该库使用GPU高效的查找表策略实现了any4以及其他常见量化方法。我们的代码已开源，详见这个链接：[这里](this https URL)。 

---
# PRIME: Large Language Model Personalization with Cognitive Memory and Thought Processes 

**Title (ZH)**: PRIME: 大型语言模型个性化设计基于认知记忆与思维过程 

**Authors**: Xinliang Frederick Zhang, Nick Beauchamp, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04607)  

**Abstract**: Large language model (LLM) personalization aims to align model outputs with individuals' unique preferences and opinions. While recent efforts have implemented various personalization methods, a unified theoretical framework that can systematically understand the drivers of effective personalization is still lacking. In this work, we integrate the well-established cognitive dual-memory model into LLM personalization, by mirroring episodic memory to historical user engagements and semantic memory to long-term, evolving user beliefs. Specifically, we systematically investigate memory instantiations and introduce a unified framework, PRIME, using episodic and semantic memory mechanisms. We further augment PRIME with a novel personalized thinking capability inspired by the slow thinking strategy. Moreover, recognizing the absence of suitable benchmarks, we introduce a dataset using Change My View (CMV) from Reddit, specifically designed to evaluate long-context personalization. Extensive experiments validate PRIME's effectiveness across both long- and short-context scenarios. Further analysis confirms that PRIME effectively captures dynamic personalization beyond mere popularity biases. 

**Abstract (ZH)**: 大型语言模型个性化旨在使模型输出与个体的独特偏好和意见相一致。虽然近期已实施了各种个性化方法，但缺乏一个能够系统理解有效个性化的驱动因素的统一理论框架。在此工作中，我们将成熟的认知双重记忆模型整合到大型语言模型个性化中，通过镜像情景记忆反映历史用户互动，并通过语义记忆反映长期演变的用户信念。具体而言，我们系统地研究了记忆实例化，并引入了一个统一框架PRIME，使用情景记忆和语义记忆机制。此外，鉴于缺乏合适的基准，我们引入了一个使用来自Reddit的Change My View (CMV)数据集，专门用于评估长上下文个性化。广泛的实验验证了PRIME在长上下文和短上下文场景中的有效性。进一步的分析证实，PRIME能够有效捕捉超越单纯流行偏差的动态个性化。 

---
# Accelerated Online Reinforcement Learning using Auxiliary Start State Distributions 

**Title (ZH)**: 使用辅助起始状态分布加速在线强化学习 

**Authors**: Aman Mehra, Alexandre Capone, Jeff Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2507.04606)  

**Abstract**: A long-standing problem in online reinforcement learning (RL) is of ensuring sample efficiency, which stems from an inability to explore environments efficiently. Most attempts at efficient exploration tackle this problem in a setting where learning begins from scratch, without prior information available to bootstrap learning. However, such approaches fail to leverage expert demonstrations and simulators that can reset to arbitrary states. These affordances are valuable resources that offer enormous potential to guide exploration and speed up learning. In this paper, we explore how a small number of expert demonstrations and a simulator allowing arbitrary resets can accelerate learning during online RL. We find that training with a suitable choice of an auxiliary start state distribution that may differ from the true start state distribution of the underlying Markov Decision Process can significantly improve sample efficiency. We find that using a notion of safety to inform the choice of this auxiliary distribution significantly accelerates learning. By using episode length information as a way to operationalize this notion, we demonstrate state-of-the-art sample efficiency on a sparse-reward hard-exploration environment. 

**Abstract (ZH)**: 在线强化学习中确保样本效率的长期问题源于有效探索环境的能力不足。大多数高效探索的尝试是在没有先前信息可供利用以加速学习的情况下，从头开始学习。然而，这些方法未能利用可以重置到任意状态的专家演示和模拟器。这些功能是宝贵的资源，具有极大的潜力来引导探索并加速学习。在本文中，我们探讨了如何通过少量专家演示和允许任意重置的模拟器来加速在线强化学习中的学习。我们发现，使用一个与底层马尔可夫决策过程的真实起始状态分布可能不同的适当辅助起始状态分布进行训练，可以显著提高样本效率。我们发现，使用安全性的概念来指导这种辅助分布的选择可以显著加快学习速度。通过使用 Episode 长度信息来实现这一概念，我们展示了在稀疏奖励和困难探索环境中达到最先进的样本效率。 

---
# Lilith: Developmental Modular LLMs with Chemical Signaling 

**Title (ZH)**: Lilith: 发展型化学信号调控的模块化大语言模型 

**Authors**: Mohid Farooqi, Alejandro Comas-Leon  

**Link**: [PDF](https://arxiv.org/pdf/2507.04575)  

**Abstract**: Current paradigms in Artificial Intelligence rely on layers of feedforward networks which model brain activity at the neuronal level. We conjecture that expanding to the level of multiple brain regions with chemical signaling may be a productive step toward understanding the emergence of consciousness. We propose LILITH, a novel architecture that combines developmental training of modular language models with brain-inspired token-based communication protocols, mirroring chemical signaling in the brain. Our approach models distinct brain regions as specialized LLM modules including thinking, memory, sensory, and regulatory components that communicate through emergent token-based signaling protocols analogous to neurotransmitter networks. Unlike traditional pre-trained systems, LILITH would employ developmental training where untrained LLM architectures learn through simulated life experiences, developing communication pathways and cognitive abilities through environmental interaction and evolutionary optimization. This framework would enable direct empirical investigation of consciousness emergence using Integrated Information Theory metrics while providing unprecedented insight into inter-module signaling patterns during development. By optimizing for consciousness emergence rather than task performance, LILITH could provide insight into different emergent phenomena at multiple levels of neural correlates, contrasting neuronal-level processing with multi-region coordination dynamics. The goal of this paper is to put the idea forward while recognizing the substantial challenges in implementing such a system. 

**Abstract (ZH)**: 当前的人工智能范式依赖于多层前馈网络来模拟神经元层面的大脑活动。我们推测，扩展到包含化学信号在内的多个脑区层次可能是一个理解意识涌现的有成效的步骤。我们提出了LILITH这一新颖架构，结合模块化语言模型的发育训练与受脑启发的基于令牌的通信协议，模拟大脑中的化学信号网络。我们的方法将不同的大脑区域建模为专业化的LLM模块，包括思维、记忆、感觉和调节组件，并通过类神经递质网络的涌现性基于令牌的信号协议进行通信。不同于传统的预训练系统，LILITH会采用发育训练方法，让未训练的LLM架构通过模拟生活经验学习，通过环境互动和进化优化发展出通讯路径和认知能力。该框架可以使用综合信息理论指标直接进行意识涌现的实证研究，同时为开发过程中的模块间信号模式提供前所未有的见解。通过优化意识涌现而非任务性能，LILITH可以为多个神经相关层面的不同涌现现象提供洞察，对比神经元级处理与多区域协调动力学。本文旨在提出这一理念，同时认识到实现这样一个系统的巨大挑战。 

---
# Nile-Chat: Egyptian Language Models for Arabic and Latin Scripts 

**Title (ZH)**: 尼罗河聊天：埃及语模型 for 阿拉伯 script 和拉丁 script 

**Authors**: Guokan Shang, Hadi Abdine, Ahmad Chamma, Amr Mohamed, Mohamed Anwar, Abdelaziz Bounhar, Omar El Herraoui, Preslav Nakov, Michalis Vazirgiannis, Eric Xing  

**Link**: [PDF](https://arxiv.org/pdf/2507.04569)  

**Abstract**: We introduce Nile-Chat-4B, 3x4B-A6B, and 12B, a collection of LLMs for Egyptian dialect, uniquely designed to understand and generate texts written in both Arabic and Latin scripts. Specifically, with Nile-Chat-3x4B-A6B, we introduce a novel language adaptation approach by leveraging the Branch-Train-MiX strategy to merge script-specialized experts, into a single MoE model. Our Nile-Chat models significantly outperform leading multilingual and Arabic LLMs, such as LLaMa, Jais, and ALLaM, on our newly introduced Egyptian evaluation benchmarks, which span both understanding and generative tasks. Notably, our 12B model yields a 14.4% performance gain over Qwen2.5-14B-Instruct on Latin-script benchmarks. All our resources are publicly available. We believe this work presents a comprehensive methodology for adapting LLMs to dual-script languages, addressing an often overlooked aspect in modern LLM development. 

**Abstract (ZH)**: 我们介绍了针对埃及方言的Nile-Chat-4B、3x4B-A6B和12B语言模型，这些模型独特地设计用于理解和生成使用阿拉伯字母和拉丁字母书写的文本。特别地，通过利用Branch-Train-MiX策略将专用于不同字母表的专家合并到单个模型中，我们提出了Nile-Chat-3x4B-A6B的新型语言适应方法。我们的Nile-Chat模型在我们新引入的涵盖理解和生成任务的埃及评估基准测试中显著优于LLaMa、Jais和ALLaM等领先多语言和阿拉伯语语言模型。值得注意的是，我们的12B模型在拉丁字母基准测试中比Qwen2.5-14B-Instruct性能提高了14.4%。所有资源均已公开。我们认为这项工作为适应双字母表语言的语言模型提供了一个全面的方法，这在现代语言模型开发中往往被忽视。 

---
# Evaluating LLMs on Real-World Forecasting Against Human Superforecasters 

**Title (ZH)**: 评估大型语言模型在现实世界预测任务中的表现——与人类超级预测家相比 

**Authors**: Janna Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04562)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their ability to forecast future events remains understudied. A year ago, large language models struggle to come close to the accuracy of a human crowd. I evaluate state-of-the-art LLMs on 464 forecasting questions from Metaculus, comparing their performance against human superforecasters. Frontier models achieve Brier scores that ostensibly surpass the human crowd but still significantly underperform a group of superforecasters. 

**Abstract (ZH)**: 大规模语言模型在预测未来事件方面的能力尚待研究：以Metaculus提供的464道预测题为例，前沿模型的表现虽然在贝叶斯评分上似乎超越了大众人类预测，但仍显著逊色于超级预测者。 

---
# SPIRA: Building an Intelligent System for Respiratory Insufficiency Detection 

**Title (ZH)**: SPIRA：构建一种呼吸不足检测的智能系统 

**Authors**: Renato Cordeiro Ferreira, Dayanne Gomes, Vitor Tamae, Francisco Wernke, Alfredo Goldman  

**Link**: [PDF](https://arxiv.org/pdf/2507.04548)  

**Abstract**: Respiratory insufficiency is a medic symptom in which a person gets a reduced amount of oxygen in the blood. This paper reports the experience of building SPIRA: an intelligent system for detecting respiratory insufficiency from voice. It compiles challenges faced in two succeeding implementations of the same architecture, summarizing lessons learned on data collection, training, and inference for future projects in similar systems. 

**Abstract (ZH)**: 呼吸不足是一种医疗症状，表现为人体血液中氧含量减少。本文报告了构建SPIRA智能系统以从语音中检测呼吸不足的经验。该报告总结了在相同架构的两次相继实现中遇到的挑战，并概述了在数据收集、训练和推理方面的经验教训，以供类似系统未来项目参考。 

---
# DP-Fusion: Token-Level Differentially Private Inference for Large Language Models 

**Title (ZH)**: DP-Fusion: 嵌入级差分隐私推理large语言模型 

**Authors**: Rushil Thareja, Preslav Nakov, Praneeth Vepakomma, Nils Lukas  

**Link**: [PDF](https://arxiv.org/pdf/2507.04531)  

**Abstract**: Large language models (LLMs) can leak sensitive information from their context through generated outputs, either accidentally or when prompted adversarially. Existing defenses that aim to preserve context privacy during inference either lack formal guarantees or suffer from a poor utility/privacy trade-off. We propose DP-Fusion, a token-level Differentially Private Inference (DPI) mechanism that provably bounds how much an LLM's outputs reveal about sensitive tokens in its context. We demonstrate DPI through the task of document privatization, where the goal is to paraphrase documents so that sensitive content (e.g., Personally Identifiable Information, PII) cannot be reliably inferred, while still preserving the overall utility of the text. This is controlled by a parameter $\epsilon$: $\epsilon=0$ hides PII entirely, while higher values trade off privacy for improved paraphrase quality. DP-Fusion works as follows: (i) partition sensitive tokens into disjoint privacy groups, (ii) run the LLM once per group, and (iii) blend the output distributions so that the final output remains within a fixed statistical distance of the baseline distribution produced when no privacy group is revealed. This approach allows fine-grained control over the privacy/utility trade-off but requires multiple LLM forward passes. 

**Abstract (ZH)**: 大型语言模型（LLMs）在其生成输出中可能会无意间或在对抗性提示下泄漏敏感信息。现有旨在推理过程中保护上下文隐私的防护措施要么缺乏正式保证，要么在实用性和隐私性之间表现不佳。我们提出了DP-Fusion，一种证明可限制大型语言模型输出揭示其上下文中敏感令牌信息量的令牌级别差分隐私推理（DPI）机制。我们通过文档 privatization 任务来展示 DPI，目标是改写文档以便无法可靠地推断出敏感内容（例如，个人可识别信息，PII），同时仍保持文本的整体实用性。这由参数 $\epsilon$ 控制：$\epsilon=0$ 完全隐藏 PII，而更高的值则以提高改写质量为代价换取更多的隐私性。DP-Fusion 机制如下：（i）将敏感令牌划分为互不相交的隐私组，（ii）对每个组运行一次大型语言模型，（iii）融合输出分布，使得最终输出与未揭示任何隐私组时产生的基准分布保持固定统计距离。此方法允许对隐私与实用性之间的权衡进行精细控制，但需要多次大型语言模型的前向传递。 

---
# Grounded Gesture Generation: Language, Motion, and Space 

**Title (ZH)**: 基于语境的手势生成：语言、动作与空间 

**Authors**: Anna Deichler, Jim O'Regan, Teo Guichoux, David Johansson, Jonas Beskow  

**Link**: [PDF](https://arxiv.org/pdf/2507.04522)  

**Abstract**: Human motion generation has advanced rapidly in recent years, yet the critical problem of creating spatially grounded, context-aware gestures has been largely overlooked. Existing models typically specialize either in descriptive motion generation, such as locomotion and object interaction, or in isolated co-speech gesture synthesis aligned with utterance semantics. However, both lines of work often treat motion and environmental grounding separately, limiting advances toward embodied, communicative agents. To address this gap, our work introduces a multimodal dataset and framework for grounded gesture generation, combining two key resources: (1) a synthetic dataset of spatially grounded referential gestures, and (2) MM-Conv, a VR-based dataset capturing two-party dialogues. Together, they provide over 7.7 hours of synchronized motion, speech, and 3D scene information, standardized in the HumanML3D format. Our framework further connects to a physics-based simulator, enabling synthetic data generation and situated evaluation. By bridging gesture modeling and spatial grounding, our contribution establishes a foundation for advancing research in situated gesture generation and grounded multimodal interaction.
Project page: this https URL 

**Abstract (ZH)**: 人类运动生成近年来取得了快速进展，但创建空间上具grounded性、情境aware性的手势的关键问题迄今已被广泛关注不足。现有模型通常专门处理描述性运动生成，如行进和物体交互，或者孤立的共声手势合成，与语义对齐。然而，这两方面的研究往往将运动和环境grounding分开处理，限制了具身、交际型代理的研究进展。为弥补这一空白，我们的工作引入了一个多模态数据集和框架，用于生成grounded手势，结合了两个关键资源：（1）一个空间上具grounded性的参考手势合成数据集，（2）一个基于VR的对话数据集，捕捉两人的对话。它们共同提供了超过7.7小时的同步运动、语音和三维场景信息，并统一使用了HumanML3D格式。我们的框架进一步与物理基础模拟器连接，使合成数据生成和情境评估成为可能。通过将手势建模与空间grounding相结合，我们的贡献为推进情境手势生成和多模态交互的研究奠定了基础。 

---
# MVL-Loc: Leveraging Vision-Language Model for Generalizable Multi-Scene Camera Relocalization 

**Title (ZH)**: MVL-Loc：利用视觉-语言模型实现可泛化的多场景相机重新定位 

**Authors**: Zhendong Xiao, Wu Wei, Shujie Ji, Shan Yang, Changhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.04509)  

**Abstract**: Camera relocalization, a cornerstone capability of modern computer vision, accurately determines a camera's position and orientation (6-DoF) from images and is essential for applications in augmented reality (AR), mixed reality (MR), autonomous driving, delivery drones, and robotic navigation. Unlike traditional deep learning-based methods that regress camera pose from images in a single scene, which often lack generalization and robustness in diverse environments, we propose MVL-Loc, a novel end-to-end multi-scene 6-DoF camera relocalization framework. MVL-Loc leverages pretrained world knowledge from vision-language models (VLMs) and incorporates multimodal data to generalize across both indoor and outdoor settings. Furthermore, natural language is employed as a directive tool to guide the multi-scene learning process, facilitating semantic understanding of complex scenes and capturing spatial relationships among objects. Extensive experiments on the 7Scenes and Cambridge Landmarks datasets demonstrate MVL-Loc's robustness and state-of-the-art performance in real-world multi-scene camera relocalization, with improved accuracy in both positional and orientational estimates. 

**Abstract (ZH)**: 多场景6-自由度相机重定位框架MVL-Loc 

---
# A validity-guided workflow for robust large language model research in psychology 

**Title (ZH)**: 基于效度指导的工作流在心理学中开展稳健的大语言模型研究 

**Authors**: Zhicheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.04491)  

**Abstract**: Large language models (LLMs) are rapidly being integrated into psychological research as research tools, evaluation targets, human simulators, and cognitive models. However, recent evidence reveals severe measurement unreliability: Personality assessments collapse under factor analysis, moral preferences reverse with punctuation changes, and theory-of-mind accuracy varies widely with trivial rephrasing. These "measurement phantoms"--statistical artifacts masquerading as psychological phenomena--threaten the validity of a growing body of research. Guided by the dual-validity framework that integrates psychometrics with causal inference, we present a six-stage workflow that scales validity requirements to research ambition--using LLMs to code text requires basic reliability and accuracy, while claims about psychological properties demand comprehensive construct validation. Researchers must (1) explicitly define their research goal and corresponding validity requirements, (2) develop and validate computational instruments through psychometric testing, (3) design experiments that control for computational confounds, (4) execute protocols with transparency, (5) analyze data using methods appropriate for non-independent observations, and (6) report findings within demonstrated boundaries and use results to refine theory. We illustrate the workflow through an example of model evaluation--"LLM selfhood"--showing how systematic validation can distinguish genuine computational phenomena from measurement artifacts. By establishing validated computational instruments and transparent practices, this workflow provides a path toward building a robust empirical foundation for AI psychology research. 

**Abstract (ZH)**: 大型语言模型（LLMs）正迅速被集成到心理学研究中作为研究工具、评价目标、人类模拟器和认知模型。然而，近期的证据揭示了严重的测量不可靠性：个性评估在因数分析中崩溃，道德偏好因标点更改而逆转，共情准确性因简单的重新表述而变异。这些“测量幽灵”——统计 artifacts 假装为心理现象——威胁着越来越多的研究的有效性。基于将心理测量学与因果推断结合的双重有效性框架，我们提出了一种六阶段工作流程，以适应研究雄心的要求——使用 LLM 编码文本需要基本可靠性和准确性，而关于心理属性的主张则需要全面的结构验证。研究人员必须（1）明确界定其研究目标和相应的有效性要求，（2）通过心理测量学测试开发和验证计算工具，（3）设计控制计算混杂因素的实验，（4）以透明的方式执行协议，（5）使用适合非独立观测的方法分析数据，并（6）在已证明的边界内报告研究发现，并使用结果来改进理论。我们通过一个模型评估示例——“LLM 自身性”——说明了这一工作流程，展示了系统验证如何区分真正的计算现象与测量 artifacts。通过建立验证的计算工具并采用透明的实践，此工作流程为构建稳健的 AI 心理学研究实证基础提供了途径。 

---
# Dealing with Uncertainty in Contextual Anomaly Detection 

**Title (ZH)**: 处理上下文异常检测中的不确定性 

**Authors**: Luca Bindini, Lorenzo Perini, Stefano Nistri, Jesse Davis, Paolo Frasconi  

**Link**: [PDF](https://arxiv.org/pdf/2507.04490)  

**Abstract**: Contextual anomaly detection (CAD) aims to identify anomalies in a target (behavioral) variable conditioned on a set of contextual variables that influence the normalcy of the target variable but are not themselves indicators of anomaly. In many anomaly detection tasks, there exist contextual variables that influence the normalcy of the target variable but are not themselves indicators of anomaly. In this work, we propose a novel framework for CAD, normalcy score (NS), that explicitly models both the aleatoric and epistemic uncertainties. Built on heteroscedastic Gaussian process regression, our method regards the Z-score as a random variable, providing confidence intervals that reflect the reliability of the anomaly assessment. Through experiments on benchmark datasets and a real-world application in cardiology, we demonstrate that NS outperforms state-of-the-art CAD methods in both detection accuracy and interpretability. Moreover, confidence intervals enable an adaptive, uncertainty-driven decision-making process, which may be very important in domains such as healthcare. 

**Abstract (ZH)**: 基于上下文的异常检测（Contextual Anomaly Detection, CAD）旨在在一组影响目标变量正常性但本身不是异常指标的上下文变量条件下，识别目标（行为）变量中的异常。在许多异常检测任务中，存在影响目标变量正常性但本身不是异常指标的上下文变量。本文提出了一种新颖的带有置信度评分（Normalcy Score, NS）的CAD框架，该框架明确模型了 aleatoric 和 epistemic 不确定性。基于异方差高斯过程回归，我们的方法将 Z-score 视为随机变量，提供反映异常评估可靠性的置信区间。通过对基准数据集和心脏病学领域的实际应用进行实验，我们证明了 NS 在检测准确性和可解释性方面均优于现有最先进的 CAD 方法。此外，置信区间使适应性的、基于不确定性的决策过程成为可能，在诸如医疗保健等领域中可能非常关键。 

---
# LoSiA: Efficient High-Rank Fine-Tuning via Subnet Localization and Optimization 

**Title (ZH)**: LoSiA: 通过子网络定位与优化实现的高效高秩微调 

**Authors**: Xujia Wang. Yunjia Qi, Bin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04487)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) methods, such as LoRA, significantly reduce the number of trainable parameters by introducing low-rank decomposition matrices. However, existing methods perform extensive matrix multiplications in domain specialization tasks, resulting in computational inefficiency and sub-optimal fine-tuning performance. Hence, we propose LoSiA(Low-Resources Subnet Integration Adaptation), an innovative method that dynamically localizes and optimizes critical parameters during the training process. Specifically, it identifies a sub-network using gradient sparsity analysis and optimizes it as the trainable target. This design enables effective high-rank adaptation by updating only the sub-network parameters, reducing the additional matrix multiplication. We also present LoSiA-Pro, a faster implementation of LoSiA, which reduces the training latency by about $27\%$ compared to LoRA. Extensive evaluations show that our method achieves minimal performance drop compared to full fine-tuning, while requiring the least training time across domain specialization and common-sense reasoning tasks. Further analysis shows that LoSiA also reduces forgetting during continued training. 

**Abstract (ZH)**: Parameter- Efficient Fine-Tuning (PEFT) 方法，如 LoRA，通过引入低秩分解矩阵显著减少可训练参数的数量。然而，现有方法在领域专业化任务中进行大量的矩阵乘法，导致计算效率低下和次优的微调性能。因此，我们提出了一种名为 LoSiA（Low-Resources Subnet Integration Adaptation）的创新方法，在训练过程中动态定位和优化关键参数。具体来说，它使用梯度稀疏性分析来确定一个子网络，并将其优化为可训练的目标。这种设计通过仅更新子网络参数来实现有效的高秩适应，减少了额外的矩阵乘法。我们还提出了 LoSiA-Pro 的更快实现版本，与 LoRA 相比，其训练延迟降低了约 27%。广泛评估表明，我们的方法在领域专业化和常识推理任务中实现了最小的性能下降，同时需要最短的训练时间。进一步分析表明，LoSiA 还可以在继续训练中减少遗忘。 

---
# Source Attribution in Retrieval-Augmented Generation 

**Title (ZH)**: 检索增强生成中的来源归属 

**Authors**: Ikhtiyor Nematov, Tarik Kalai, Elizaveta Kuzmenko, Gabriele Fugagnoli, Dimitris Sacharidis, Katja Hose, Tomer Sagi  

**Link**: [PDF](https://arxiv.org/pdf/2507.04480)  

**Abstract**: While attribution methods, such as Shapley values, are widely used to explain the importance of features or training data in traditional machine learning, their application to Large Language Models (LLMs), particularly within Retrieval-Augmented Generation (RAG) systems, is nascent and challenging. The primary obstacle is the substantial computational cost, where each utility function evaluation involves an expensive LLM call, resulting in direct monetary and time expenses. This paper investigates the feasibility and effectiveness of adapting Shapley-based attribution to identify influential retrieved documents in RAG. We compare Shapley with more computationally tractable approximations and some existing attribution methods for LLM. Our work aims to: (1) systematically apply established attribution principles to the RAG document-level setting; (2) quantify how well SHAP approximations can mirror exact attributions while minimizing costly LLM interactions; and (3) evaluate their practical explainability in identifying critical documents, especially under complex inter-document relationships such as redundancy, complementarity, and synergy. This study seeks to bridge the gap between powerful attribution techniques and the practical constraints of LLM-based RAG systems, offering insights into achieving reliable and affordable RAG explainability. 

**Abstract (ZH)**: 尽管Shapley值等归因方法在传统机器学习中广泛用于解释特征或训练数据的重要性，但在大型语言模型（LLMs），特别是在检索增强生成（RAG）系统中的应用仍处于起步阶段并面临挑战。主要障碍是高昂的计算成本，每次效用函数评估都涉及昂贵的LLM调用，导致直接的金钱和时间支出。本文探讨了将基于Shapley值的归因方法适应RAG中的检索文档以识别有影响力文档的可能性和有效性。我们将Shapley值与更易于计算的近似方法及现有的LLM归因方法进行比较。本文旨在：（1）系统地将现有归因原则应用于RAG文档级别设置；（2）量化Shapley值近似方法如何准确反映精确归因，同时尽量减少昂贵的LLM交互；（3）评估其在识别关键文档方面的实用解释性，尤其是在冗余、互补和协同作用等复杂文档关系下的表现。本研究旨在弥合强大归因技术与基于LLM的RAG系统实践约束之间的差距，为实现可靠且经济实惠的RAG解释性提供见解。 

---
# Model Inversion Attacks on Llama 3: Extracting PII from Large Language Models 

**Title (ZH)**: 针对LLaMA 3的模型反转攻击：从大型语言模型提取个人信息 

**Authors**: Sathesh P.Sivashanmugam  

**Link**: [PDF](https://arxiv.org/pdf/2507.04478)  

**Abstract**: Large language models (LLMs) have transformed natural language processing, but their ability to memorize training data poses significant privacy risks. This paper investigates model inversion attacks on the Llama 3.2 model, a multilingual LLM developed by Meta. By querying the model with carefully crafted prompts, we demonstrate the extraction of personally identifiable information (PII) such as passwords, email addresses, and account numbers. Our findings highlight the vulnerability of even smaller LLMs to privacy attacks and underscore the need for robust defenses. We discuss potential mitigation strategies, including differential privacy and data sanitization, and call for further research into privacy-preserving machine learning techniques. 

**Abstract (ZH)**: 大型语言模型(LLMs)已变革自然语言处理，但其记忆训练数据的能力带来了显著的隐私风险。本文探讨了对Meta开发的多语言LLM Llama 3.2进行模型反转攻击的情况。通过使用精心设计的提示查询该模型，我们展示了提取个人可识别信息(PII)，如密码、电子邮件地址和账户号码的过程。我们的研究结果强调了即使是较小的LLM也容易遭受隐私攻击，同时也突显了需要加强防护的必要性。我们讨论了潜在的缓解策略，包括差分隐私和数据 sanitization，并呼吁进一步研究隐私保护的机器学习技术。 

---
# The role of large language models in UI/UX design: A systematic literature review 

**Title (ZH)**: 大型语言模型在UI/UX设计中的作用：一项系统文献综述 

**Authors**: Ammar Ahmed, Ali Shariq Imran  

**Link**: [PDF](https://arxiv.org/pdf/2507.04469)  

**Abstract**: This systematic literature review examines the role of large language models (LLMs) in UI/UX design, synthesizing findings from 38 peer-reviewed studies published between 2022 and 2025. We identify key LLMs in use, including GPT-4, Gemini, and PaLM, and map their integration across the design lifecycle, from ideation to evaluation. Common practices include prompt engineering, human-in-the-loop workflows, and multimodal input. While LLMs are reshaping design processes, challenges such as hallucination, prompt instability, and limited explainability persist. Our findings highlight LLMs as emerging collaborators in design, and we propose directions for the ethical, inclusive, and effective integration of these technologies. 

**Abstract (ZH)**: 系统文献综述：大型语言模型在UI/UX设计中的角色研究——基于2022年至2025年间38篇同行评审论文的综合分析 

---
# The Joys of Categorical Conformal Prediction 

**Title (ZH)**: 范畴 conformal 预测的乐趣 

**Authors**: Michele Caprio  

**Link**: [PDF](https://arxiv.org/pdf/2507.04441)  

**Abstract**: Conformal prediction (CP) is an Uncertainty Representation technique that delivers finite-sample calibrated prediction regions for any underlying Machine Learning model, yet its status as an Uncertainty Quantification (UQ) tool has remained conceptually opaque. We adopt a category-theoretic approach to CP -- framing it as a morphism, embedded in a commuting diagram, of two newly-defined categories -- that brings us three joys. First, we show that -- under minimal assumptions -- CP is intrinsically a UQ mechanism, that is, its UQ capabilities are a structural feature of the method. Second, we demonstrate that CP bridges (and perhaps subsumes) the Bayesian, frequentist, and imprecise probabilistic approaches to predictive statistical reasoning. Finally, we show that a conformal prediction region (CPR) is the image of a covariant functor. This observation is relevant to AI privacy: It implies that privacy noise added locally does not break coverage. 

**Abstract (ZH)**: 同构预测（CP）是一种不确定性表示技术，能够为任何基础机器学习模型提供有限样本校准预测区域，然而其作为不确定性量化（UQ）工具的地位仍然概念上不够透明。我们采用范畴论的方法来定义CP——将其视为嵌入在两个 newly-defined 范畴中的态射，并带来了三个喜悦。首先，我们证明在最小假设下，CP 内在地是一种UQ机制，即其UQ能力是该方法的结构特征。其次，我们展示了CP连接（并且可能子囊括）贝叶斯、频率主义和不精确概率预测统计推理的方法论。最后，我们证明同构预测区域（CPR）是共变函子的象。这一观察对于AI隐私具有重要意义：它表明局部添加的隐私噪声不会破坏覆盖性。 

---
# Learning Software Bug Reports: A Systematic Literature Review 

**Title (ZH)**: 学习软件 bug 报告：一项系统文献综述 

**Authors**: Guoming Long, Jingzhi Gong, Hui Fang, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.04422)  

**Abstract**: The recent advancement of artificial intelligence, especially machine learning (ML), has significantly impacted software engineering research, including bug report analysis. ML aims to automate the understanding, extraction, and correlation of information from bug reports. Despite its growing importance, there has been no comprehensive review in this area. In this paper, we present a systematic literature review covering 1,825 papers, selecting 204 for detailed analysis. We derive seven key findings: 1) Extensive use of CNN, LSTM, and $k$NN for bug report analysis, with advanced models like BERT underutilized due to their complexity. 2) Word2Vec and TF-IDF are popular for feature representation, with a rise in deep learning approaches. 3) Stop word removal is the most common preprocessing, with structural methods rising after 2020. 4) Eclipse and Mozilla are the most frequently evaluated software projects. 5) Bug categorization is the most common task, followed by bug localization and severity prediction. 6) There is increasing attention on specific bugs like non-functional and performance bugs. 7) Common evaluation metrics are F1-score, Recall, Precision, and Accuracy, with $k$-fold cross-validation preferred for model evaluation. 8) Many studies lack robust statistical tests. We also identify six promising future research directions to provide useful insights for practitioners. 

**Abstract (ZH)**: 最近人工智能的进步，尤其是机器学习（ML），显著影响了软件工程研究，包括错误报告分析。机器学习旨在自动化理解、提取和关联错误报告中的信息。尽管其重要性日益凸显，但该领域尚未进行过全面回顾。在本文中，我们进行了系统文献综述，共涵盖1,825篇论文，并选择204篇进行详细分析。我们得出了七个关键发现：1) 广泛使用CNN、LSTM和$k$NN进行错误报告分析，但因复杂性问题，高级模型如BERT的应用不足。2) Word2Vec和TF-IDF广泛用于特征表示，深度学习方法呈上升趋势。3) 去除停用词是最常见的预处理方法，自2020年起，结构化方法逐渐增多。4) Eclipse和Mozilla是最常评估的软件项目。5) 错误分类是最常见的任务，其次是错误定位和严重性预测。6) 不断关注特定类型的错误，如非功能性错误和性能错误。7) 常见的评估指标包括F1分数、召回率、精确率和准确率，$k$-折交叉验证常用于模型评估。8) 很多研究缺乏稳健的统计检验。我们还指出了六个有前景的未来研究方向，为实践者提供了有价值的见解。 

---
# Multimedia Verification Through Multi-Agent Deep Research Multimodal Large Language Models 

**Title (ZH)**: 多代理深度研究驱动的多媒体验证通过多模态大型语言模型 

**Authors**: Huy Hoan Le, Van Sy Thinh Nguyen, Thi Le Chi Dang, Vo Thanh Khang Nguyen, Truong Thanh Hung Nguyen, Hung Cao  

**Link**: [PDF](https://arxiv.org/pdf/2507.04410)  

**Abstract**: This paper presents our submission to the ACMMM25 - Grand Challenge on Multimedia Verification. We developed a multi-agent verification system that combines Multimodal Large Language Models (MLLMs) with specialized verification tools to detect multimedia misinformation. Our system operates through six stages: raw data processing, planning, information extraction, deep research, evidence collection, and report generation. The core Deep Researcher Agent employs four tools: reverse image search, metadata analysis, fact-checking databases, and verified news processing that extracts spatial, temporal, attribution, and motivational context. We demonstrate our approach on a challenge dataset sample involving complex multimedia content. Our system successfully verified content authenticity, extracted precise geolocation and timing information, and traced source attribution across multiple platforms, effectively addressing real-world multimedia verification scenarios. 

**Abstract (ZH)**: 本文呈现了我们参加ACMMM25多媒体验证大赛的提交内容。我们开发了一个结合多模态大型语言模型和专门验证工具的多代理验证系统，以检测多媒体错误信息。该系统通过六 stages：原始数据处理、计划、信息提取、深度研究、证据收集和报告生成。核心深度研究员代理使用了四种工具：反向图像搜索、元数据分析、事实核查数据库以及提取空间、时间、归属和动机上下文的可信新闻处理。我们利用一个包含复杂多媒体内容的挑战数据集样本展示了我们的方法。我们的系统成功验证了内容的真实性，提取了精确的地理定位和时间信息，并跨多个平台追踪了来源归属，有效应对了实际的多媒体验证场景。 

---
# SpiritRAG: A Q&A System for Religion and Spirituality in the United Nations Archive 

**Title (ZH)**: SpiritRAG：联合国档案中宗教与灵性领域的问答系统 

**Authors**: Yingqiang Gao, Fabian Winiger, Patrick Montjourides, Anastassia Shaitarova, Nianlong Gu, Simon Peng-Keller, Gerold Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2507.04395)  

**Abstract**: Religion and spirituality (R/S) are complex and highly domain-dependent concepts which have long confounded researchers and policymakers. Due to their context-specificity, R/S are difficult to operationalize in conventional archival search strategies, particularly when datasets are very large, poorly accessible, and marked by information noise. As a result, considerable time investments and specialist knowledge is often needed to extract actionable insights related to R/S from general archival sources, increasing reliance on published literature and manual desk reviews. To address this challenge, we present SpiritRAG, an interactive Question Answering (Q&A) system based on Retrieval-Augmented Generation (RAG). Built using 7,500 United Nations (UN) resolution documents related to R/S in the domains of health and education, SpiritRAG allows researchers and policymakers to conduct complex, context-sensitive database searches of very large datasets using an easily accessible, chat-based web interface. SpiritRAG is lightweight to deploy and leverages both UN documents and user provided documents as source material. A pilot test and evaluation with domain experts on 100 manually composed questions demonstrates the practical value and usefulness of SpiritRAG. 

**Abstract (ZH)**: 宗教与灵性（R/S）是复杂且高度领域依赖的概念，长久以来困扰着研究人员和政策制定者。由于其情境特异性，R/S在常规档案检索策略中难以操作化，尤其是在数据集非常庞大、访问受限且充满信息噪声的情况下。因此，往往需要大量时间和专家知识来从通用档案资源中提取与R/S相关的可操作洞见，增加了对已出版文献和手工桌面审查的依赖。为应对这一挑战，我们提出了基于检索增强生成（RAG）的交互式问答系统SpiritRAG。该系统基于7,500份与健康和教育领域宗教与灵性相关的联合国决议文件构建，允许研究人员和政策制定者通过易于访问的基于聊天的网络界面进行复杂的情境敏感数据库搜索。SpiritRAG轻量级且易于部署，利用联合国文件和用户提供的文件作为源材料。在100个手动编制的问题上进行的试点测试和专家评估表明，SpiritRAG具有实际价值和实用性。 

---
# Tractable Representation Learning with Probabilistic Circuits 

**Title (ZH)**: 可计算的概率电路中的可计算表示学习 

**Authors**: Steven Braun, Sahil Sidheekh, Antonio Vergari, Martin Mundt, Sriraam Natarajan, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2507.04385)  

**Abstract**: Probabilistic circuits (PCs) are powerful probabilistic models that enable exact and tractable inference, making them highly suitable for probabilistic reasoning and inference tasks. While dominant in neural networks, representation learning with PCs remains underexplored, with prior approaches relying on external neural embeddings or activation-based encodings. To address this gap, we introduce autoencoding probabilistic circuits (APCs), a novel framework leveraging the tractability of PCs to model probabilistic embeddings explicitly. APCs extend PCs by jointly modeling data and embeddings, obtaining embedding representations through tractable probabilistic inference. The PC encoder allows the framework to natively handle arbitrary missing data and is seamlessly integrated with a neural decoder in a hybrid, end-to-end trainable architecture enabled by differentiable sampling. Our empirical evaluation demonstrates that APCs outperform existing PC-based autoencoding methods in reconstruction quality, generate embeddings competitive with, and exhibit superior robustness in handling missing data compared to neural autoencoders. These results highlight APCs as a powerful and flexible representation learning method that exploits the probabilistic inference capabilities of PCs, showing promising directions for robust inference, out-of-distribution detection, and knowledge distillation. 

**Abstract (ZH)**: 概率电路（PCs）是强大的概率模型，能够实现精确且可计算的推理，使其非常适合概率推理和推理任务。尽管在神经网络中占据主导地位，但使用PCs进行表示学习仍处于起步阶段，此前的方法依赖于外部神经嵌入或基于激活的编码。为解决这一问题，我们引入了自编码概率电路（APCs），这是一种新颖的框架，利用概率电路的计算性来显式建模概率嵌入。APCs通过联合建模数据和嵌入，通过可计算的概率推理获得嵌入表示。PC编码器使框架能够原生处理任意缺失数据，并通过可微采样在混合的端到端可训练架构中无缝集成神经解码器。我们的实证评估表明，APCs在重构质量上优于现有的基于PCs的自编码方法，生成的嵌入与神经自编码器具有竞争力，并且在处理缺失数据的鲁棒性方面表现出更优越的性能。这些结果突显了APCs作为一种强大且灵活的表示学习方法的角色，它利用了概率电路的推理能力，并为稳健推理、异常检测和知识蒸馏等方向提供了有前景的方向。 

---
# Transferring Visual Explainability of Self-Explaining Models through Task Arithmetic 

**Title (ZH)**: 通过任务算术转移自我解释模型的视觉可解释性 

**Authors**: Yuya Yoshikawa, Ryotaro Shimizu, Takahiro Kawashima, Yuki Saito  

**Link**: [PDF](https://arxiv.org/pdf/2507.04380)  

**Abstract**: In scenarios requiring both prediction and explanation efficiency for image classification, self-explaining models that perform both tasks in a single inference are effective. However, their training incurs substantial labeling and computational costs. This study aims to tackle the issue by proposing a method to transfer the visual explainability of self-explaining models, learned in a source domain, to a target domain based on a task arithmetic framework. Specifically, we construct a self-explaining model by extending image classifiers based on a vision-language pretrained model. We then define an \emph{explainability vector} as the difference between model parameters trained on the source domain with and without explanation supervision. Based on the task arithmetic framework, we impart explainability to a model trained only on the prediction task in the target domain by applying the explainability vector. Experimental results on various image classification datasets demonstrate that, except for transfers between some less-related domains, visual explainability can be successfully transferred from source to target domains, improving explanation quality in the target domain without sacrificing classification accuracy. Furthermore, we show that the explainability vector learned on a large and diverse dataset like ImageNet, extended with explanation supervision, exhibits universality and robustness, improving explanation quality on nine out of ten different target datasets. We also find that the explanation quality achieved with a single model inference is comparable to that of Kernel SHAP, which requires 150 model inferences. 

**Abstract (ZH)**: 在既要求预测又要求解释效率的图像分类场景中，能够同时完成两项任务的自解释模型效果显著，但其训练过程会带来显著的标注和计算成本。本研究旨在通过提出一种方法，将源自源域的自解释模型的视觉可解释性转移到目标域，基于任务算术框架实现这一目标。具体而言，我们通过在视觉-语言预训练模型基础上扩展图像分类器来构建一个自解释模型。然后，定义一个可解释性向量为在源域中带有和不带有解释监督训练的模型参数之间的差异。基于任务算术框架，我们通过对仅在目标域上进行预测任务训练的模型应用可解释性向量，赋予其可解释性。在多个图像分类数据集上的实验结果表明，除了某些较不相关的领域之间的转移外，自解释模型的视觉可解释性可以从源域成功转移到目标域，提高目标域的解释质量而不牺牲分类准确率。此外，我们在ImageNet等大型多样数据集上学习并加入解释监督的可解释性向量表现出普遍性和鲁棒性，在十个不同目标数据集中有九个上显著提高了解释质量。我们还发现，通过单次模型推理实现的解释质量与需要150次模型推理的Kernel SHAP相当。 

---
# Attention Slipping: A Mechanistic Understanding of Jailbreak Attacks and Defenses in LLMs 

**Title (ZH)**: 注意力泄露：LLM中劫持攻击及其防御机制的理解 

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho  

**Link**: [PDF](https://arxiv.org/pdf/2507.04365)  

**Abstract**: As large language models (LLMs) become more integral to society and technology, ensuring their safety becomes essential. Jailbreak attacks exploit vulnerabilities to bypass safety guardrails, posing a significant threat. However, the mechanisms enabling these attacks are not well understood. In this paper, we reveal a universal phenomenon that occurs during jailbreak attacks: Attention Slipping. During this phenomenon, the model gradually reduces the attention it allocates to unsafe requests in a user query during the attack process, ultimately causing a jailbreak. We show Attention Slipping is consistent across various jailbreak methods, including gradient-based token replacement, prompt-level template refinement, and in-context learning. Additionally, we evaluate two defenses based on query perturbation, Token Highlighter and SmoothLLM, and find they indirectly mitigate Attention Slipping, with their effectiveness positively correlated with the degree of mitigation achieved. Inspired by this finding, we propose Attention Sharpening, a new defense that directly counters Attention Slipping by sharpening the attention score distribution using temperature scaling. Experiments on four leading LLMs (Gemma2-9B-It, Llama3.1-8B-It, Qwen2.5-7B-It, Mistral-7B-It v0.2) show that our method effectively resists various jailbreak attacks while maintaining performance on benign tasks on AlpacaEval. Importantly, Attention Sharpening introduces no additional computational or memory overhead, making it an efficient and practical solution for real-world deployment. 

**Abstract (ZH)**: 大型语言模型（LLMs）在社会和技术中的作用日益重要，确保其安全性变得至关重要。脱牢笼攻击通过利用漏洞来绕过安全防护，构成了重大威胁。然而，这些攻击机制尚不完全清楚。本文揭示了脱牢笼攻击过程中普遍存在的一种现象：注意力溜逸。在此过程中，模型会逐渐减少对攻击性请求的注意力分配，最终导致脱牢笼。我们证明注意力溜逸在各种脱牢笼方法（包括基于梯度的标记替换、提示级模板精炼和上下文学习）中是一致存在的。此外，我们评估了两种基于查询扰动的防御措施——Token Highlighter和SmoothLLM，并发现它们间接减轻了注意力溜逸，其效果与减轻程度正相关。受此发现启发，我们提出了一种新的防御方法——注意力强化，通过使用温度缩放直接对抗注意力溜逸以增强注意力分数分布。实验表明，我们的方法能够有效地抵御各种脱牢笼攻击，同时在AlpacaEval上的良性任务上保持性能。重要的是，注意力强化不会引入额外的计算或内存开销，使其成为一个高效且实用的现实部署解决方案。 

---
# Mission-Aligned Learning-Informed Control of Autonomous Systems: Formulation and Foundations 

**Title (ZH)**: 自主系统的目标导向学习驱动控制：形式化与基础理论 

**Authors**: Vyacheslav Kungurtsev, Gustav Sir, Akhil Anand, Sebastien Gros, Haozhe Tian, Homayoun Hamedmoghadam  

**Link**: [PDF](https://arxiv.org/pdf/2507.04356)  

**Abstract**: Research, innovation and practical capital investment have been increasing rapidly toward the realization of autonomous physical agents. This includes industrial and service robots, unmanned aerial vehicles, embedded control devices, and a number of other realizations of cybernetic/mechatronic implementations of intelligent autonomous devices. In this paper, we consider a stylized version of robotic care, which would normally involve a two-level Reinforcement Learning procedure that trains a policy for both lower level physical movement decisions as well as higher level conceptual tasks and their sub-components. In order to deliver greater safety and reliability in the system, we present the general formulation of this as a two-level optimization scheme which incorporates control at the lower level, and classical planning at the higher level, integrated with a capacity for learning. This synergistic integration of multiple methodologies -- control, classical planning, and RL -- presents an opportunity for greater insight for algorithm development, leading to more efficient and reliable performance. Here, the notion of reliability pertains to physical safety and interpretability into an otherwise black box operation of autonomous agents, concerning users and regulators. This work presents the necessary background and general formulation of the optimization framework, detailing each component and its integration with the others. 

**Abstract (ZH)**: 研究、创新和实际资本投资正迅速增加以实现自主物理代理。这包括工业和服务业机器人、无人驾驶航空车辆、嵌入式控制设备以及许多其他基于信息技术和机电一体化的智能自主设备。本文考虑了一种精简版的机器人护理，通常涉及两层强化学习程序，用于训练低层物理动作决策和高层概念任务及其子组件的策略。为了提高系统的安全性和可靠性，我们提出将这一过程作为包含低层控制和高层经典规划的两层优化方案，结合学习能力。多种方法——控制、经典规划和RL——的这种协同集成为算法开发提供了更多的洞察力，从而实现更高效和可靠的性能。在这里，可靠性的概念涉及物理安全和对自主代理黑盒操作的可解释性，对于用户和监管机构而言。本文提供了优化框架的必要背景和一般性形式，并详细说明了每个组成部分及其与其他组成部分的集成。 

---
# AI-washing: The Asymmetric Effects of Its Two Types on Consumer Moral Judgments 

**Title (ZH)**: AI清洗：两种类型对其消费者道德判断的不对称影响 

**Authors**: Greg Nyilasy, Harsha Gangadharbatla  

**Link**: [PDF](https://arxiv.org/pdf/2507.04352)  

**Abstract**: As AI hype continues to grow, organizations face pressure to broadcast or downplay purported AI initiatives - even when contrary to truth. This paper introduces AI-washing as overstating (deceptive boasting) or understating (deceptive denial) a company's real AI usage. A 2x2 experiment (N = 401) examines how these false claims affect consumer attitudes and purchase intentions. Results reveal a pronounced asymmetry: deceptive denial evokes more negative moral judgments than honest negation, while deceptive boasting has no effects. We show that perceived betrayal mediates these outcomes. By clarifying how AI-washing erodes trust, the study highlights clear ethical implications for policymakers, marketers, and researchers striving for transparency. 

**Abstract (ZH)**: 随着AI hype的不断增长，组织面临宣传或淡化所谓AI倡议的压力——即使这与事实相反。本文介绍了AI-washing，即夸大（欺骗性吹嘘）或缩小（欺骗性否认）公司实际AI使用情况。一项2x2实验（N=401）探讨了这些虚假声明如何影响消费者态度和购买意向。结果揭示了一个显著的不对称性：欺骗性否认比诚实否认引发更强烈的负面道德判断，而欺骗性吹嘘则没有影响。我们展示了感知到的背叛在这些结果中起中介作用。通过阐明AI-washing如何损害信任，该研究强调了透明度追求者——政策制定者、营销人员和研究人员——的明确伦理影响。 

---
# MLLM-Fabric: Multimodal Large Language Model-Driven Robotic Framework for Fabric Sorting and Selection 

**Title (ZH)**: MLLM-Fabric：多模态大规模语言模型驱动的纺织品分拣与选择机器人框架 

**Authors**: Liman Wang, Hanyang Zhong, Tianyuan Wang, Shan Luo, Jihong Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04351)  

**Abstract**: Choosing the right fabric is crucial to meet functional and quality requirements in robotic applications for textile manufacturing, apparel production, and smart retail. We present MLLM-Fabric, a robotic framework powered by multimodal large language models (MLLMs) for fabric sorting and selection. The system includes a robotic arm, a camera, a visuotactile sensor, and a pressure sensor. It employs supervised fine-tuning and multimodal explanation-guided knowledge distillation to accurately classify and rank fabric properties. To facilitate further research, we release a dataset of 220 unique fabric samples, including RGB images and synchronized visuotactile and pressure data. Experimental results show that our Fabric-Llama-90B model consistently outperforms pretrained vision-language baselines in both property ranking accuracy and selection reliability. 

**Abstract (ZH)**: 选择合适的织物对于纺织制造、服装生产及智能零售中的机器人应用至关重要。我们提出了一种基于多模态大型语言模型（MLLMs）的机器人框架MLLM-Fabric，用于织物分类和选择。该系统包括机械臂、摄像头、视触觉传感器和压力传感器。它采用监督微调和多模态解释引导的知识 distillation 技术，准确分类和排序织物属性。为促进进一步研究，我们发布了包含220种独特织物样本的数据集，其中包括RGB图像和同步的视触觉及压力数据。实验结果表明，我们的Fabric-Llama-90B模型在属性排序准确性和选择可靠性方面均优于预训练的视觉-语言基线模型。 

---
# Improving Action Smoothness for a Cascaded Online Learning Flight Control System 

**Title (ZH)**: 改进级联在线学习飞行控制系统的动作平滑性 

**Authors**: Yifei Li, Erik-jan van Kampen  

**Link**: [PDF](https://arxiv.org/pdf/2507.04346)  

**Abstract**: This paper aims to improve the action smoothness of a cascaded online learning flight control system. Although the cascaded structure is widely used in flight control design, its stability can be compromised by oscillatory control actions, which poses challenges for practical engineering applications. To address this issue, we introduce an online temporal smoothness technique and a low-pass filter to reduce the amplitude and frequency of the control actions. Fast Fourier Transform (FFT) is used to analyze policy performance in the frequency domain. Simulation results demonstrate the improvements achieved by the two proposed techniques. 

**Abstract (ZH)**: 本文旨在改善级联在线学习飞行控制系统中的动作平滑度。虽然级联结构在飞行控制设计中广泛应用，但其稳定性可能因振荡控制动作而受损，这给实际工程应用带来了挑战。为解决这一问题，我们引入了一种在线时间域平滑技术及低通滤波器来降低控制动作的幅度和频率。快速傅里叶变换（FFT）用于在频域分析策略性能。仿真实验结果展示了所提出的两种技术所取得的改进。 

---
# Efficient Perplexity Bound and Ratio Matching in Discrete Diffusion Language Models 

**Title (ZH)**: 离散扩散语言模型中的高效困惑度边界与比率匹配 

**Authors**: Etrit Haxholli, Yeti Z. Gürbüz, Oğul Can, Eli Waxman  

**Link**: [PDF](https://arxiv.org/pdf/2507.04341)  

**Abstract**: While continuous diffusion models excel in modeling continuous distributions, their application to categorical data has been less effective. Recent work has shown that ratio-matching through score-entropy within a continuous-time discrete Markov chain (CTMC) framework serves as a competitive alternative to autoregressive models in language modeling. To enhance this framework, we first introduce three new theorems concerning the KL divergence between the data and learned distribution. Our results serve as the discrete counterpart to those established for continuous diffusion models and allow us to derive an improved upper bound of the perplexity. Second, we empirically show that ratio-matching performed by minimizing the denoising cross-entropy between the clean and corrupted data enables models to outperform those utilizing score-entropy with up to 10% lower perplexity/generative-perplexity, and 15% faster training steps. To further support our findings, we introduce and evaluate a novel CTMC transition-rate matrix that allows prediction refinement, and derive the analytic expression for its matrix exponential which facilitates the computation of conditional ratios thus enabling efficient training and generation. 

**Abstract (ZH)**: 虽然连续扩散模型在建模连续分布方面表现出色，但它们对分类数据的应用效果不佳。最近的研究表明，通过连续时间离散马尔可夫链（CTMC）框架内的得分-熵比值匹配，可以作为一种与自回归模型在语言建模中竞争的替代方案。为了增强该框架，我们首先介绍了关于数据与学习分布之间KL散度的三个新定理。我们的结果是连续扩散模型已建立结果的离散对应部分，并使我们能够推导出困惑度改进的上界。其次，实验证明，通过最小化清洁数据和受污染数据之间的去噪交叉熵来进行的比值匹配，可以使模型在困惑度/生成困惑度降低高达10%以及训练步骤加快15%的情况下优于使用得分-熵的方法。为进一步支持我们的发现，我们引入并评估了一种新的CTMC转换率矩阵，该矩阵允许预测细化，并推导出其矩阵指数的解析表达式，从而便于条件比值的计算，从而实现高效的训练和生成。 

---
# CLIP-RL: Surgical Scene Segmentation Using Contrastive Language-Vision Pretraining & Reinforcement Learning 

**Title (ZH)**: CLIP-RL: 使用对比言语-视觉预训练与强化学习的手术场景分割 

**Authors**: Fatmaelzahraa Ali Ahmed, Muhammad Arsalan, Abdulaziz Al-Ali, Khalid Al-Jalham, Shidin Balakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2507.04317)  

**Abstract**: Understanding surgical scenes can provide better healthcare quality for patients, especially with the vast amount of video data that is generated during MIS. Processing these videos generates valuable assets for training sophisticated models. In this paper, we introduce CLIP-RL, a novel contrastive language-image pre-training model tailored for semantic segmentation for surgical scenes. CLIP-RL presents a new segmentation approach which involves reinforcement learning and curriculum learning, enabling continuous refinement of the segmentation masks during the full training pipeline. Our model has shown robust performance in different optical settings, such as occlusions, texture variations, and dynamic lighting, presenting significant challenges. CLIP model serves as a powerful feature extractor, capturing rich semantic context that enhances the distinction between instruments and tissues. The RL module plays a pivotal role in dynamically refining predictions through iterative action-space adjustments. We evaluated CLIP-RL on the EndoVis 2018 and EndoVis 2017 datasets. CLIP-RL achieved a mean IoU of 81%, outperforming state-of-the-art models, and a mean IoU of 74.12% on EndoVis 2017. This superior performance was achieved due to the combination of contrastive learning with reinforcement learning and curriculum learning. 

**Abstract (ZH)**: 理解手术场景可以提高患者的医疗质量，尤其是在微创手术（MIS）过程中生成的大量视频数据中。处理这些视频可以生成用于训练高级模型的宝贵资源。在本文中，我们介绍了CLIP-RL，这是一种专为手术场景语义分割设计的新颖对比语言-图像预训练模型。CLIP-RL提出了一种新的分割方法，该方法结合了强化学习和递增学习，使分割掩码在完整训练管道中得到持续优化。我们的模型在不同的光学设置下，如遮挡、纹理变化和动态光照等具有挑战性的场景中表现出稳健的性能。CLIP模型作为强大的特征提取器，捕获了丰富的语义上下文，增强了仪器和组织之间的区分。RL模块在通过迭代的动作空间调整中动态优化预测中起到了关键作用。我们在EndoVis 2018和EndoVis 2017数据集上评估了CLIP-RL。CLIP-RL在EndoVis 2018上的平均IoU为81%，在EndoVis 2017上为74.12%，这一优越性能得益于对比学习、强化学习和递增学习的结合。 

---
# Surg-SegFormer: A Dual Transformer-Based Model for Holistic Surgical Scene Segmentation 

**Title (ZH)**: Surg-SegFormer: 一种基于双变换器的整域手术场景分割模型 

**Authors**: Fatimaelzahraa Ahmed, Muraam Abdel-Ghani, Muhammad Arsalan, Mahmoud Ali, Abdulaziz Al-Ali, Shidin Balakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2507.04304)  

**Abstract**: Holistic surgical scene segmentation in robot-assisted surgery (RAS) enables surgical residents to identify various anatomical tissues, articulated tools, and critical structures, such as veins and vessels. Given the firm intraoperative time constraints, it is challenging for surgeons to provide detailed real-time explanations of the operative field for trainees. This challenge is compounded by the scarcity of expert surgeons relative to trainees, making the unambiguous delineation of go- and no-go zones inconvenient. Therefore, high-performance semantic segmentation models offer a solution by providing clear postoperative analyses of surgical procedures. However, recent advanced segmentation models rely on user-generated prompts, rendering them impractical for lengthy surgical videos that commonly exceed an hour. To address this challenge, we introduce Surg-SegFormer, a novel prompt-free model that outperforms current state-of-the-art techniques. Surg-SegFormer attained a mean Intersection over Union (mIoU) of 0.80 on the EndoVis2018 dataset and 0.54 on the EndoVis2017 dataset. By providing robust and automated surgical scene comprehension, this model significantly reduces the tutoring burden on expert surgeons, empowering residents to independently and effectively understand complex surgical environments. 

**Abstract (ZH)**: 机器人辅助手术中的整体手术场景分割使外科居民能够识别各种解剖组织、机械设备以及关键结构（如静脉和血管）。鉴于手术过程中的严格时间限制，外科医生难以为学员提供详细的实时术野解释。由于专家外科医生的数量相对较少，这一挑战更加复杂，使得明确划定可行区和不可行区变得不便。因此，高性能的语义分割模型提供了解决方案，通过提供术后清晰的手术过程分析。然而，近期先进的分割模型依赖于用户生成的提示，使得它们在常见的超一小时的手术视频中不实用。为了解决这一挑战，我们提出了Surg-SegFormer，这是一种全新的无提示模型，其性能 outranks 当前最先进的技术。Surg-SegFormer 在 EndoVis2018 数据集上的平均交并比（mIoU）为 0.80，在 EndoVis2017 数据集上的 mIoU 为 0.54。通过提供 robust 和自动化的手术场景理解，该模型显著减轻了专家外科医生的指导负担，使居民能够独立有效地理解复杂的手术环境。 

---
# QF: Quick Feedforward AI Model Training without Gradient Back Propagation 

**Title (ZH)**: QF：无需梯度反向传播的快速前向AI模型训练 

**Authors**: Feng Qi  

**Link**: [PDF](https://arxiv.org/pdf/2507.04300)  

**Abstract**: We propose Quick Feedforward (QF) Learning, a novel knowledge consolidation framework for transformer-based models that enables efficient transfer of instruction derived knowledge into model weights through feedforward activations without any gradient back propagation. Unlike traditional finetuning, QF updates are computed in closed form, require minimal parameter modification, and preserve prior knowledge. Importantly, QF allows models to train and infer within the same runtime environment, making the process more resource efficient and closely aligned with how the human brain operates. Code and models are open sourced on GitHub. I hope QF Learning inspires a more efficient and brain-like paradigm for AI systems. 

**Abstract (ZH)**: 快速前馈学习：transformer基模型的一种新型知识 consolidation框架 

---
# LearnLens: LLM-Enabled Personalised, Curriculum-Grounded Feedback with Educators in the Loop 

**Title (ZH)**: LearnLens: 由教育者参与的基于课程内容的个性化LLM反馈 

**Authors**: Runcong Zhao, Artem Borov, Jiazheng Li, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2507.04295)  

**Abstract**: Effective feedback is essential for student learning but is time-intensive for teachers. We present LearnLens, a modular, LLM-based system that generates personalised, curriculum-aligned feedback in science education. LearnLens comprises three components: (1) an error-aware assessment module that captures nuanced reasoning errors; (2) a curriculum-grounded generation module that uses a structured, topic-linked memory chain rather than traditional similarity-based retrieval, improving relevance and reducing noise; and (3) an educator-in-the-loop interface for customisation and oversight. LearnLens addresses key challenges in existing systems, offering scalable, high-quality feedback that empowers both teachers and students. 

**Abstract (ZH)**: 有效的反馈对于学生学习至关重要，但对学生而言耗时较多。我们提出LearnLens，这是一种模块化、基于大语言模型的系统，用于生成个性化且与课程内容对齐的科学教育反馈。LearnLens由三个部分组成：（1）一个错误感知评估模块，能够捕捉细微的推理错误；（2）一个基于课程内容的生成模块，使用结构化的、主题关联的记忆链而非传统的基于相似性的检索，从而提高相关性和减少噪声；（3）一个教师在环路的接口，用于个性化定制和监督。LearnLens解决了现有系统的关键挑战，提供可扩展、高质量的反馈，赋能教师和学生。 

---
# M$^3$-Med: A Benchmark for Multi-lingual, Multi-modal, and Multi-hop Reasoning in Medical Instructional Video Understanding 

**Title (ZH)**: M$^3$-Med：多语言、多模态和多跳推理在医学教学视频理解中的基准测试 

**Authors**: Shenxi Liu, Kan Li, Mingyang Zhao, Yuhang Tian, Bin Li, Shoujun Zhou, Hongliang Li, Fuxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04289)  

**Abstract**: With the rapid progress of artificial intelligence (AI) in multi-modal understanding, there is increasing potential for video comprehension technologies to support professional domains such as medical education. However, existing benchmarks suffer from two primary limitations: (1) Linguistic Singularity: they are largely confined to English, neglecting the need for multilingual resources; and (2) Shallow Reasoning: their questions are often designed for surface-level information retrieval, failing to properly assess deep multi-modal integration. To address these limitations, we present M3-Med, the first benchmark for Multi-lingual, Multi-modal, and Multi-hop reasoning in Medical instructional video understanding. M3-Med consists of medical questions paired with corresponding video segments, annotated by a team of medical experts. A key innovation of M3-Med is its multi-hop reasoning task, which requires a model to first locate a key entity in the text, then find corresponding visual evidence in the video, and finally synthesize information across both modalities to derive the answer. This design moves beyond simple text matching and poses a substantial challenge to a model's deep cross-modal understanding capabilities. We define two tasks: Temporal Answer Grounding in Single Video (TAGSV) and Temporal Answer Grounding in Video Corpus (TAGVC). We evaluated several state-of-the-art models and Large Language Models (LLMs) on M3-Med. The results reveal a significant performance gap between all models and human experts, especially on the complex multi-hop questions where model performance drops sharply. M3-Med effectively highlights the current limitations of AI models in deep cross-modal reasoning within specialized domains and provides a new direction for future research. 

**Abstract (ZH)**: 多语言、多模态和多跳推理的医疗指令视频理解基准（M3-Med） 

---
# SeqTex: Generate Mesh Textures in Video Sequence 

**Title (ZH)**: SeqTex: 生成视频序列中的网格纹理 

**Authors**: Ze Yuan, Xin Yu, Yangtian Sun, Yuan-Chen Guo, Yan-Pei Cao, Ding Liang, Xiaojuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2507.04285)  

**Abstract**: Training native 3D texture generative models remains a fundamental yet challenging problem, largely due to the limited availability of large-scale, high-quality 3D texture datasets. This scarcity hinders generalization to real-world scenarios. To address this, most existing methods finetune foundation image generative models to exploit their learned visual priors. However, these approaches typically generate only multi-view images and rely on post-processing to produce UV texture maps -- an essential representation in modern graphics pipelines. Such two-stage pipelines often suffer from error accumulation and spatial inconsistencies across the 3D surface. In this paper, we introduce SeqTex, a novel end-to-end framework that leverages the visual knowledge encoded in pretrained video foundation models to directly generate complete UV texture maps. Unlike previous methods that model the distribution of UV textures in isolation, SeqTex reformulates the task as a sequence generation problem, enabling the model to learn the joint distribution of multi-view renderings and UV textures. This design effectively transfers the consistent image-space priors from video foundation models into the UV domain. To further enhance performance, we propose several architectural innovations: a decoupled multi-view and UV branch design, geometry-informed attention to guide cross-domain feature alignment, and adaptive token resolution to preserve fine texture details while maintaining computational efficiency. Together, these components allow SeqTex to fully utilize pretrained video priors and synthesize high-fidelity UV texture maps without the need for post-processing. Extensive experiments show that SeqTex achieves state-of-the-art performance on both image-conditioned and text-conditioned 3D texture generation tasks, with superior 3D consistency, texture-geometry alignment, and real-world generalization. 

**Abstract (ZH)**: 利用预训练视频基础模型直接生成完整的UV纹理图的SeqTex端到端框架 

---
# VOLTRON: Detecting Unknown Malware Using Graph-Based Zero-Shot Learning 

**Title (ZH)**: VOLTRON: 使用图为基础的零样本学习检测未知恶意软件 

**Authors**: M. Tahir Akdeniz, Zeynep Yeşilkaya, İ. Enes Köse, İ. Ulaş Ünal, Sevil Şen  

**Link**: [PDF](https://arxiv.org/pdf/2507.04275)  

**Abstract**: The persistent threat of Android malware presents a serious challenge to the security of millions of users globally. While many machine learning-based methods have been developed to detect these threats, their reliance on large labeled datasets limits their effectiveness against emerging, previously unseen malware families, for which labeled data is scarce or nonexistent.
To address this challenge, we introduce a novel zero-shot learning framework that combines Variational Graph Auto-Encoders (VGAE) with Siamese Neural Networks (SNN) to identify malware without needing prior examples of specific malware families. Our approach leverages graph-based representations of Android applications, enabling the model to detect subtle structural differences between benign and malicious software, even in the absence of labeled data for new threats.
Experimental results show that our method outperforms the state-of-the-art MaMaDroid, especially in zero-day malware detection. Our model achieves 96.24% accuracy and 95.20% recall for unknown malware families, highlighting its robustness against evolving Android threats. 

**Abstract (ZH)**: 持久存在的Android恶意软件威胁对全球亿万用户的安全构成了严重挑战。虽然已经开发出许多基于机器学习的方法来检测这些威胁，但它们依赖于大型标注数据集，这限制了它们对新兴的、之前未见过的恶意软件家族的有效性，这类恶意软件的标注数据稀缺或不存在。
为应对这一挑战，我们提出了一种新颖的零样本学习框架，该框架结合了变分图自编码器（VGAE）和双面神经网络（SNN），以无需特定恶意软件家族的先例即可识别恶意软件。该方法利用Android应用程序的图基表示，使模型能够在缺乏新威胁标注数据的情况下，检测良性软件和恶意软件之间的细微结构差异。
实验结果表明，我们的方法在未知恶意软件家族的检测方面优于最先进的MaMaDroid，特别是在零日恶意软件检测方面表现优异。我们的模型对未知恶意软件家族的准确率和召回率分别为96.24%和95.20%，突显了其对抗演化的Android威胁的稳定性。 

---
# ZERO: Multi-modal Prompt-based Visual Grounding 

**Title (ZH)**: ZERO: 多模态提示驱动的视觉定位 

**Authors**: Sangbum Choi, Kyeongryeol Go  

**Link**: [PDF](https://arxiv.org/pdf/2507.04270)  

**Abstract**: Recent advances in artificial intelligence have led to the emergence of foundation models, large-scale pre-trained neural networks that serve as versatile starting points for a wide range of downstream tasks. In this work, we present ZERO, a zero-shot multi-prompt object detection model specifically designed for robust, production-ready deployment across diverse industrial domains. ZERO integrates direct image input with multiple user-defined prompts, which can include both textual and visual cues, and processes them through dedicated encoders to generate accurate detection outputs. The model architecture is optimized for scalability, with a total of 1.033 TFLOPS and 622.346 million parameters, and is trained using a domain-specific image database exceeding one billion images. For the CVPR 2025 Foundational Few-Shot Object Detection (FSOD) Challenge, we introduce a domain-specific fine-tuning strategy that emphasizes prompt diversity and conservative pseudo-labeling, enabling effective adaptation to new domains with minimal supervision. Our approach demonstrates practical advantages in flexibility, efficiency, and real-world applicability, achieving strong performance on the RF20VL-fsod benchmark despite limited annotation budgets. The results highlight the potential of prompt-driven, data-centric AI for scalable and adaptive object detection in dynamic industrial environments. 

**Abstract (ZH)**: 近期人工智能的进展催生了基础模型，即大规模预训练神经网络，这些模型可以作为广泛下游任务的多功能起点。在本工作中，我们提出了ZERO，这是一种零样本多提示对象检测模型，专门设计用于在多样化的工业领域中进行稳健的生产部署。ZERO结合了直接图像输入与多个用户定义的提示，这些提示可以包括文本和视觉线索，并通过专用编码器生成准确的检测输出。模型架构针对可扩展性进行了优化，总共有1.033 TFLOPS和622.346百万参数，并使用包含超过十亿张图像的领域专用图像数据库进行训练。在CVPR 2025基础少量样本对象检测（FSOD）挑战中，我们提出了一种领域专用的微调策略，强调提示多样性并采用保守的伪标签方法，以在最少监督的情况下有效适应新领域。我们的方法在灵活性、效率和实际应用性方面表现出实际优势，尽管在注释预算有限的情况下仍取得了出色性能。结果突显了以提示驱动的数据为中心的人工智能在动态工业环境中的可扩展和适应性对象检测潜力。 

---
# Deep-Learning-Assisted Highly-Accurate COVID-19 Diagnosis on Lung Computed Tomography Images 

**Title (ZH)**: 深度学习辅助high精度肺癌计算机断层扫描图像COVID-19诊断 

**Authors**: Yinuo Wang, Juhyun Bae, Ka Ho Chow, Shenyang Chen, Shreyash Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2507.04252)  

**Abstract**: COVID-19 is a severe and acute viral disease that can cause symptoms consistent with pneumonia in which inflammation is caused in the alveolous regions of the lungs leading to a build-up of fluid and breathing difficulties. Thus, the diagnosis of COVID using CT scans has been effective in assisting with RT-PCR diagnosis and severity classifications. In this paper, we proposed a new data quality control pipeline to refine the quality of CT images based on GAN and sliding windows. Also, we use class-sensitive cost functions including Label Distribution Aware Loss(LDAM Loss) and Class-balanced(CB) Loss to solve the long-tail problem existing in datasets. Our model reaches more than 0.983 MCC in the benchmark test dataset. 

**Abstract (ZH)**: COVID-19是一种严重的急性病毒疾病，可导致与肺炎一致的症状，炎症发生在肺泡区域，导致液体积聚和呼吸困难。因此，使用CT扫描诊断COVID-19已有效辅助RT-PCR诊断和病情分类。本文提出了一种基于GAN和滑动窗口的新数据质量控制管道，以 refinement CT图像的质量。同时，我们使用包括标签分布感知损失(LDAM Loss)和类别平衡(CB)损失在内的类敏感成本函数来解决数据集中存在的长尾问题。在基准测试数据集中，我们的模型达到了超过0.983的MCC。 

---
# Just Enough Shifts: Mitigating Over-Refusal in Aligned Language Models with Targeted Representation Fine-Tuning 

**Title (ZH)**: 刚刚好多少次迁移：通过目标导向的表示微调减轻对齐语言模型的过度拒绝问题 

**Authors**: Mahavir Dabas, Si Chen, Charles Fleming, Ming Jin, Ruoxi Jia  

**Link**: [PDF](https://arxiv.org/pdf/2507.04250)  

**Abstract**: Safety alignment is crucial for large language models (LLMs) to resist malicious instructions but often results in over-refusals, where benign prompts are unnecessarily rejected, impairing user experience and model utility. We introduce ACTOR (Activation-Based Training for Over-Refusal Reduction), a robust and compute- and data-efficient training framework that minimizes over-refusals by leveraging internal activation patterns from diverse queries. ACTOR precisely identifies and adjusts the activation components that trigger refusals, providing stronger control over the refusal mechanism. By fine-tuning only a single model layer, ACTOR effectively reduces over-refusals across multiple benchmarks while maintaining the model's ability to handle harmful queries and preserve overall utility. 

**Abstract (ZH)**: 基于激活的训练框架ACTOR：减少过度拒绝以提高大语言模型的安全性和实用性 

---
# Domain Generalizable Portrait Style Transfer 

**Title (ZH)**: 通用域人物风格迁移 

**Authors**: Xinbo Wang, Wenju Xu, Qing Zhang, Wei-Shi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.04243)  

**Abstract**: This paper presents a portrait style transfer method that generalizes well to various different domains while enabling high-quality semantic-aligned stylization on regions including hair, eyes, eyelashes, skins, lips, and background. To this end, we propose to establish dense semantic correspondence between the given input and reference portraits based on a pre-trained model and a semantic adapter, with which we obtain a warped reference semantically aligned with the input. To ensure effective yet controllable style transfer, we devise an AdaIN-Wavelet transform to balance content preservation and stylization by blending low-frequency information of the warped reference with high-frequency information of the input in the latent space. A style adapter is also designed to provide style guidance from the warped reference. With the stylized latent from AdaIN-Wavelet transform, we employ a dual-conditional diffusion model that integrates a ControlNet recording high-frequency information and the style guidance to generate the final result. Extensive experiments demonstrate the superiority of our method. Our code and trained model are available at this https URL. 

**Abstract (ZH)**: 本文提出了一种通用性良好的人物风格迁移方法，能够在保持高质量语义对齐的前提下，对包括头发、眼睛、眼睑、皮肤、嘴唇和背景等多个区域进行风格化。为此，我们提出基于预训练模型和语义适配器建立给定输入和参考人物的密集语义对应关系，从而获得一个语义上与输入对齐的变形参考图像。为了确保风格迁移的有效性与可控性，我们设计了一种AdaIN-Wavelet 变换，在潜在空间中通过融合变形参考图像的低频信息和输入的高频信息来平衡内容保持与风格化。此外，我们还设计了一个风格适配器来提供来自变形参考图像的风格指导。使用AdaIN-Wavelet 变换生成的风格化潜在表示，我们采用一个双条件扩散模型，结合高频率信息记录模块ControlNet和风格指导，生成最终结果。大量实验验证了该方法的优势。我们的代码和训练模型可从以下链接获取。 

---
# Scaling Context Requires Rethinking Attention 

**Title (ZH)**: 扩大上下文需要重新思考注意力机制。 

**Authors**: Carles Gelada, Jacob Buckman, Sean Zhang, Txus Bach  

**Link**: [PDF](https://arxiv.org/pdf/2507.04239)  

**Abstract**: We argue that neither transformers nor sub-quadratic architectures are well suited to training at long sequence lengths: the cost of processing the context is too expensive in the former, too inexpensive in the latter. Approaches such as sliding window attention which reduce the cost-per-token of a transformer impair in-context learning, and so are also unsuitable. To address these limitations, we introduce power attention, an architectural layer for linear-cost sequence modeling whose state size can be adjusted independently of parameters, unlocking the advantages of linear attention on practical domains. We develop and open-source a set of GPU kernels for efficient power attention, identifying a novel pattern of operation fusion to avoid memory and bandwidth bottlenecks. Our experiments on the in-context learning of power attention shows that these models dominate both exponential attention and linear attention at long-context training. 

**Abstract (ZH)**: 我们argue认为，transformers和次二次复杂度的架构都不适合在长序列长度下进行训练：前者处理上下文的成本太高，后者则太低。诸如滑动窗口注意力等方法虽然能够降低每词处理成本，但会损害上下文学习，因此也不合适。为了解决这些问题，我们引入了幂次注意力，这是一种具有线性成本的序列建模架构层，其状态大小可以独立于参数调整，从而在实际应用中充分发挥线性注意力的优势。我们开发并开源了一套高效幂次注意力的GPU内核，并发现了一种新的操作融合模式来避免内存和带宽瓶颈。我们的实验表明，在长上下文训练中，幂次注意力模型在上下文学习方面优于指数注意力和线性注意力。 

---
# Design Optimization of Three-Dimensional Wire Arrangement Considering Wire Crossings for Tendon-driven Robots 

**Title (ZH)**: 考虑导线交叉的 tendon 驱动机器人三维导线布局设计优化 

**Authors**: Kento Kawaharazuka, Shintaro Inoue, Yuta Sahara, Keita Yoneda, Temma Suzuki, Kei Okada  

**Link**: [PDF](https://arxiv.org/pdf/2507.04235)  

**Abstract**: Tendon-driven mechanisms are useful from the perspectives of variable stiffness, redundant actuation, and lightweight design, and they are widely used, particularly in hands, wrists, and waists of robots. The design of these wire arrangements has traditionally been done empirically, but it becomes extremely challenging when dealing with complex structures. Various studies have attempted to optimize wire arrangement, but many of them have oversimplified the problem by imposing conditions such as restricting movements to a 2D plane, keeping the moment arm constant, or neglecting wire crossings. Therefore, this study proposes a three-dimensional wire arrangement optimization that takes wire crossings into account. We explore wire arrangements through a multi-objective black-box optimization method that ensures wires do not cross while providing sufficient joint torque along a defined target trajectory. For a 3D link structure, we optimize the wire arrangement under various conditions, demonstrate its effectiveness, and discuss the obtained design solutions. 

**Abstract (ZH)**: 腱驱动机制从变量刚度、冗余驱动和轻量化设计的角度非常有用，并且在机器人手部、腕部和腰部等部位得到了广泛应用。这些绳索布置的传统设计通常是经验性的，但处理复杂结构时变得极其具有挑战性。尽管已有许多研究试图优化绳索布置，但其中许多研究通过施加限制条件，如将运动限制在2D平面、保持力矩臂不变或忽略绳索交叉，从而简化了问题。因此，本研究提出了一种考虑绳索交叉的三维绳索布置优化方法。我们通过多目标黑盒优化方法探索绳索布置，确保绳索不交叉同时在预定目标轨迹上提供足够的关节扭矩。对于三维链结构，我们在各种条件下优化了绳索布置，展示了其有效性并讨论了获得的设计方案。 

---
# High-Resolution Sustain Pedal Depth Estimation from Piano Audio Across Room Acoustics 

**Title (ZH)**: 高分辨率钢琴踏板深度估计跨房间声学环境中的钢琴音频 

**Authors**: Kun Fang, Hanwen Zhang, Ziyu Wang, Ichiro Fujinaga  

**Link**: [PDF](https://arxiv.org/pdf/2507.04230)  

**Abstract**: Piano sustain pedal detection has previously been approached as a binary on/off classification task, limiting its application in real-world piano performance scenarios where pedal depth significantly influences musical expression. This paper presents a novel approach for high-resolution estimation that predicts continuous pedal depth values. We introduce a Transformer-based architecture that not only matches state-of-the-art performance on the traditional binary classification task but also achieves high accuracy in continuous pedal depth estimation. Furthermore, by estimating continuous values, our model provides musically meaningful predictions for sustain pedal usage, whereas baseline models struggle to capture such nuanced expressions with their binary detection approach. Additionally, this paper investigates the influence of room acoustics on sustain pedal estimation using a synthetic dataset that includes varied acoustic conditions. We train our model with different combinations of room settings and test it in an unseen new environment using a "leave-one-out" approach. Our findings show that the two baseline models and ours are not robust to unseen room conditions. Statistical analysis further confirms that reverberation influences model predictions and introduces an overestimation bias. 

**Abstract (ZH)**: 基于变压器的高分辨率踏板深度估计方法及室声学影响研究 

---
# Hijacking JARVIS: Benchmarking Mobile GUI Agents against Unprivileged Third Parties 

**Title (ZH)**: 劫持JARVIS：评估移动GUI代理软件对无特权第三方的安全性 

**Authors**: Guohong Liu, Jialei Ye, Jiacheng Liu, Yuanchun Li, Wei Liu, Pengzhi Gao, Jian Luan, Yunxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04227)  

**Abstract**: Mobile GUI agents are designed to autonomously execute diverse device-control tasks by interpreting and interacting with mobile screens. Despite notable advancements, their resilience in real-world scenarios where screen content may be partially manipulated by untrustworthy third parties remains largely unexplored. Owing to their black-box and autonomous nature, these agents are vulnerable to manipulations that could compromise user devices. In this work, we present the first systematic investigation into the vulnerabilities of mobile GUI agents. We introduce a scalable attack simulation framework AgentHazard, which enables flexible and targeted modifications of screen content within existing applications. Leveraging this framework, we develop a comprehensive benchmark suite comprising both a dynamic task execution environment and a static dataset of vision-language-action tuples, totaling over 3,000 attack scenarios. The dynamic environment encompasses 58 reproducible tasks in an emulator with various types of hazardous UI content, while the static dataset is constructed from 210 screenshots collected from 14 popular commercial apps. Importantly, our content modifications are designed to be feasible for unprivileged third parties. We evaluate 7 widely-used mobile GUI agents and 5 common backbone models using our benchmark. Our findings reveal that all examined agents are significantly influenced by misleading third-party content (with an average misleading rate of 28.8% in human-crafted attack scenarios) and that their vulnerabilities are closely linked to the employed perception modalities and backbone LLMs. Furthermore, we assess training-based mitigation strategies, highlighting both the challenges and opportunities for enhancing the robustness of mobile GUI agents. Our code and data will be released at this https URL. 

**Abstract (ZH)**: 移动GUI代理的漏洞系统性研究：AgentHazard框架下的攻击模拟与基准测试 

---
# Zero-Shot Cyclic Peptide Design with Composable Geometric Conditions 

**Title (ZH)**: 基于可组合几何条件的零样本环肽设计 

**Authors**: Dapeng Jiang, Xiangzhe Kong, Jiaqi Han, Mingyu Li, Rui Jiao, Wenbing Huang, Stefano Ermon, Jianzhu Ma, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04225)  

**Abstract**: Cyclic peptides, characterized by geometric constraints absent in linear peptides, offer enhanced biochemical properties, presenting new opportunities to address unmet medical needs. However, designing target-specific cyclic peptides remains underexplored due to limited training data. To bridge the gap, we propose CP-Composer, a novel generative framework that enables zero-shot cyclic peptide generation via composable geometric constraints. Our approach decomposes complex cyclization patterns into unit constraints, which are incorporated into a diffusion model through geometric conditioning on nodes and edges. During training, the model learns from unit constraints and their random combinations in linear peptides, while at inference, novel constraint combinations required for cyclization are imposed as input. Experiments show that our model, despite trained with linear peptides, is capable of generating diverse target-binding cyclic peptides, reaching success rates from 38% to 84% on different cyclization strategies. 

**Abstract (ZH)**: 周期肽因其几何约束不同于线性肽，展现出增强的生物化学性能，为满足未满足的医疗需求提供了新的机会。然而，由于训练数据有限，设计特定靶标的周期肽仍是一个未充分探索的领域。为克服这一挑战，我们提出了一种新颖的生成框架CP-Composer，它可以利用组合几何约束实现零样本周期肽的生成。我们的方法将复杂的环化模式分解为单元约束，并通过节点和边的几何条件将这些单元约束整合到扩散模型中。在训练过程中，模型从线性肽中单元约束及其随机组合中学习，而在推理过程中，为环化所需的新型约束组合被用作输入。实验结果显示，尽管模型仅使用线性肽进行训练，仍能够生成多种目标结合的周期肽，不同环化策略的成功率范围从38%到84%。 

---
# Fairness Evaluation of Large Language Models in Academic Library Reference Services 

**Title (ZH)**: 学术图书馆参考服务中大型语言模型的公平性评估 

**Authors**: Haining Wang, Jason Clark, Yueru Yan, Star Bradley, Ruiyang Chen, Yiqiong Zhang, Hengyi Fu, Zuoyu Tian  

**Link**: [PDF](https://arxiv.org/pdf/2507.04224)  

**Abstract**: As libraries explore large language models (LLMs) for use in virtual reference services, a key question arises: Can LLMs serve all users equitably, regardless of demographics or social status? While they offer great potential for scalable support, LLMs may also reproduce societal biases embedded in their training data, risking the integrity of libraries' commitment to equitable service. To address this concern, we evaluate whether LLMs differentiate responses across user identities by prompting six state-of-the-art LLMs to assist patrons differing in sex, race/ethnicity, and institutional role. We found no evidence of differentiation by race or ethnicity, and only minor evidence of stereotypical bias against women in one model. LLMs demonstrated nuanced accommodation of institutional roles through the use of linguistic choices related to formality, politeness, and domain-specific vocabularies, reflecting professional norms rather than discriminatory treatment. These findings suggest that current LLMs show a promising degree of readiness to support equitable and contextually appropriate communication in academic library reference services. 

**Abstract (ZH)**: 图书馆探索大型语言模型在虚拟参考服务中的应用时，一个关键问题浮现：大型语言模型能否公平地服务于所有用户，不论其种族、社会经济地位等背景？ 

---
# Context Tuning for In-Context Optimization 

**Title (ZH)**: 上下文调整以实现上下文优化 

**Authors**: Jack Lu, Ryan Teehan, Zhenbang Yang, Mengye Ren  

**Link**: [PDF](https://arxiv.org/pdf/2507.04221)  

**Abstract**: We introduce Context Tuning, a simple and effective method to significantly enhance few-shot adaptation of language models (LLMs) without fine-tuning model parameters. While prompt-based adaptation techniques have demonstrated the effectiveness of lightweight adaptation methods for large language models (LLMs), they typically initialize a trainable prompt or prefix with irrelevant tokens for the task at hand. In contrast, Context Tuning initializes the trainable prompt or prefix with task-specific demonstration examples, leveraging the model's inherent In-Context Learning (ICL) ability to extract relevant information for improved few-shot learning performance. Extensive evaluations on benchmarks such as CrossFit, UnifiedQA, MMLU, BIG-Bench Hard, and ARC demonstrate that Context Tuning outperforms traditional prompt-based adaptation methods and achieves competitive accuracy to Test-Time Training with significantly higher training efficiency. 

**Abstract (ZH)**: Context Tuning: 一种简单有效的Few-Shot语言模型适应方法 

---
# Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs 

**Title (ZH)**: 模型坍缩不仅是大型语言模型机器遗忘中的一个bug，而是其一个特征 

**Authors**: Yan Scholten, Sophie Xhonneux, Stephan Günnemann, Leo Schwinn  

**Link**: [PDF](https://arxiv.org/pdf/2507.04219)  

**Abstract**: Current unlearning methods for LLMs optimize on the private information they seek to remove by incorporating it into their training objectives. We argue this not only risks reinforcing exposure to sensitive data, it also fundamentally contradicts the principle of minimizing its use. As a remedy, we propose a novel unlearning method - Partial Model Collapse (PMC), which does not require unlearning targets in the unlearning objective. Our approach is inspired by recent observations that training generative models on their own generations leads to distribution collapse, effectively removing information from the model. Our core idea is to leverage this collapse for unlearning by triggering collapse partially on the sensitive data. We theoretically analyze that our approach converges to the desired outcome, i.e. the LLM unlearns the information in the forget set. We empirically demonstrate that PMC overcomes two key limitations of existing unlearning approaches that explicitly optimize on unlearning targets, and more effectively removes private information from model outputs. Overall, our contributions represent an important step toward more comprehensive unlearning that aligns with real-world privacy constraints. Code available at this https URL. 

**Abstract (ZH)**: 当前的大规模语言模型去学习方法通过将其欲移除的私人信息纳入训练目标来优化，这不仅增加了敏感数据曝光的风险，还从根本上违背了最小化使用该信息的原则。为解决这一问题，我们提出了一种新型的去学习方法——部分模型塌陷（Partial Model Collapse，PMC），该方法不需要在去学习目标中指定去学习的目标。我们的方法受到最近观察启发，即在生成模型上使用其自身的生成结果会导致分布塌陷，有效地从模型中移除信息。我们的核心思想是利用这种塌陷来进行去学习，通过部分触发模型在敏感数据上的塌陷来实现。我们从理论上分析了该方法能够达到期望的结果，即大规模语言模型从记忆集中移除信息。我们还通过实验展示了PMC克服了现有显式优化去学习目标方法的两个关键限制，更有效移除了模型输出中的私人信息。总体而言，我们的贡献代表着朝着更符合实际隐私约束的全面去学习迈出的重要一步。代码可在以下链接获得：this https URL。 

---
# Mixed-Sample SGD: an End-to-end Analysis of Supervised Transfer Learning 

**Title (ZH)**: 混合样本SGD：监督迁移学习的端到端分析 

**Authors**: Yuyang Deng, Samory Kpotufe  

**Link**: [PDF](https://arxiv.org/pdf/2507.04194)  

**Abstract**: Theoretical works on supervised transfer learning (STL) -- where the learner has access to labeled samples from both source and target distributions -- have for the most part focused on statistical aspects of the problem, while efficient optimization has received less attention. We consider the problem of designing an SGD procedure for STL that alternates sampling between source and target data, while maintaining statistical transfer guarantees without prior knowledge of the quality of the source data. A main algorithmic difficulty is in understanding how to design such an adaptive sub-sampling mechanism at each SGD step, to automatically gain from the source when it is informative, or bias towards the target and avoid negative transfer when the source is less informative.
We show that, such a mixed-sample SGD procedure is feasible for general prediction tasks with convex losses, rooted in tracking an abstract sequence of constrained convex programs that serve to maintain the desired transfer guarantees.
We instantiate these results in the concrete setting of linear regression with square loss, and show that the procedure converges, with $1/\sqrt{T}$ rate, to a solution whose statistical performance on the target is adaptive to the a priori unknown quality of the source. Experiments with synthetic and real datasets support the theory. 

**Abstract (ZH)**: 监督迁移学习的理论研究——统计方面的工作占据了主导地位，而高效的优化方法则相对较少受到关注 

---
# SymbolicThought: Integrating Language Models and Symbolic Reasoning for Consistent and Interpretable Human Relationship Understanding 

**Title (ZH)**: 符号思维：将语言模型与符号推理结合以实现一致和可解释的人际关系理解 

**Authors**: Runcong Zhao, Qinglin Zhu, Hainiu Xu, Bin Liang, Yulan He, Lin Gui  

**Link**: [PDF](https://arxiv.org/pdf/2507.04189)  

**Abstract**: Understanding character relationships is essential for interpreting complex narratives and conducting socially grounded AI research. However, manual annotation is time-consuming and low in coverage, while large language models (LLMs) often produce hallucinated or logically inconsistent outputs. We present SymbolicThought, a human-in-the-loop framework that combines LLM-based extraction with symbolic reasoning. The system constructs editable character relationship graphs, refines them using seven types of logical constraints, and enables real-time validation and conflict resolution through an interactive interface. To support logical supervision and explainable social analysis, we release a dataset of 160 interpersonal relationships with corresponding logical structures. Experiments show that SymbolicThought improves annotation accuracy and consistency while significantly reducing time cost, offering a practical tool for narrative understanding, explainable AI, and LLM evaluation. 

**Abstract (ZH)**: 理解角色关系对于解读复杂叙事和开展社会导向的AI研究至关重要。然而，人工标注耗时且覆盖率低，而大型语言模型（LLMs）通常会产生虚假信息或逻辑不一致的输出。我们提出了一种人机结合框架SymbolicThought，该框架结合了基于LLM的提取与符号推理。该系统构建可编辑的角色关系图，通过七种类型的逻辑约束对其进行细化，并通过交互界面实现实时验证和冲突解决。为支持逻辑监督和可解释的社会分析，我们发布了包含160个人际关系及其相应逻辑结构的数据集。实验表明，SymbolicThought在提高标注准确性和一致性的同时，显著减少了时间成本，提供了一个用于叙事理解、可解释AI和LLM评估的实用工具。 

---
# Uncertainty Quantification in the Tsetlin Machine 

**Title (ZH)**: Tsetlin机中的不确定性量化 

**Authors**: Runar Helin, Ole-Christoffer Granmo, Mayur Kishor Shende, Lei Jiao, Vladimir I. Zadorozhny, Kunal Ganesh Dumbre, Rishad Shafik, Alex Yakovlev  

**Link**: [PDF](https://arxiv.org/pdf/2507.04175)  

**Abstract**: Data modeling using Tsetlin machines (TMs) is all about building logical rules from the data features. The decisions of the model are based on a combination of these logical rules. Hence, the model is fully transparent and it is possible to get explanations of its predictions. In this paper, we present a probability score for TM predictions and develop new techniques for uncertainty quantification to increase the explainability further. The probability score is an inherent property of any TM variant and is derived through an analysis of the TM learning dynamics. Simulated data is used to show a clear connection between the learned TM probability scores and the underlying probabilities of the data. A visualization of the probability scores also reveals that the TM is less confident in its predictions outside the training data domain, which contrasts the typical extrapolation phenomenon found in Artificial Neural Networks. The paper concludes with an application of the uncertainty quantification techniques on an image classification task using the CIFAR-10 dataset, where they provide new insights and suggest possible improvements to current TM image classification models. 

**Abstract (ZH)**: 基于Tsetlin机的数据建模涉及从数据特征中构建逻辑规则。模型的决策基于这些逻辑规则的组合。因此，该模型完全透明，可以解释其预测。本文为此，我们提出了一种Tsetlin机预测的概率分数，并开发了新的不确定性量化技术以进一步提高可解释性。概率分数是任何Tsetlin机变体的固有属性，通过分析Tsetlin机学习动力学获得。我们使用模拟数据来明确地展示了所学的Tsetlin机概率分数与数据底层概率之间的联系。概率分数的可视化还表明，Tsetlin机在其训练数据域之外的预测不太有信心，这与人工神经网络中典型的外推现象形成对比。最后，我们提出了一种不确定性量化技术在CIFAR-10图像分类任务中的应用，这些技术提供了新的见解并建议了当前Tsetlin机图像分类模型的可能改进。 

---
# Structure As Search: Unsupervised Permutation Learning for Combinatorial Optimization 

**Title (ZH)**: 结构即搜索：无监督排列学习在组合优化中的应用 

**Authors**: Yimeng Min, Carla P. Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2507.04164)  

**Abstract**: We propose a non-autoregressive framework for the Travelling Salesman Problem where solutions emerge directly from learned permutations without explicit search. By applying a similarity transformation to Hamiltonian cycles, the model learns to approximate permutation matrices via continuous relaxations. Our unsupervised approach achieves competitive performance against classical heuristics, demonstrating that the inherent structure of the problem can effectively guide combinatorial optimization without sequential decision-making. 

**Abstract (ZH)**: 我们提出了一种非自回归框架来解决旅行商问题，其中解决方案直接从学习到的排列中涌现出来，无需显式的搜索。通过将哈密顿循环应用相似变换，模型通过连续松弛学习近似排列矩阵。我们的无监督方法在与经典启发式方法的竞争中表现出色，证明了问题固有的结构可以有效引导组合优化，而无需序列决策。 

---
# Physics-informed neural networks and neural operators for a study of EUV electromagnetic wave diffraction from a lithography mask 

**Title (ZH)**: 基于物理的神经网络和神经运算子研究极紫外线电磁波透过掩膜后的衍射现象 

**Authors**: Vasiliy A. Es'kin, Egor V. Ivanov  

**Link**: [PDF](https://arxiv.org/pdf/2507.04153)  

**Abstract**: Physics-informed neural networks (PINNs) and neural operators (NOs) for solving the problem of diffraction of Extreme Ultraviolet (EUV) electromagnetic waves from a mask are presented. A novel hybrid Waveguide Neural Operator (WGNO) is introduced, which is based on a waveguide method with its most computationally expensive part replaced by a neural network. Numerical experiments on realistic 2D and 3D masks show that the WGNO achieves state-of-the-art accuracy and inference time, providing a highly efficient solution for accelerating the design workflows of lithography masks. 

**Abstract (ZH)**: 基于物理的神经网络（PINNs）和神经算子（NOs）在解决极端紫外线（EUV）电磁波从掩模衍射问题中的应用：一种新颖的波导神经算子（WGNO）的引入及其在 realistic 2D 和 3D 掩模上的数值实验表明，WGNO 达到了最先进的准确性和推理时间，为光刻掩模的设计工作流程加速提供了高效解决方案。 

---
# Dissecting Clinical Reasoning in Language Models: A Comparative Study of Prompts and Model Adaptation Strategies 

**Title (ZH)**: 语言模型中临床推理分解：提示和模型适应策略的比较研究 

**Authors**: Mael Jullien, Marco Valentino, Leonardo Ranaldi, Andre Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2507.04142)  

**Abstract**: Recent works on large language models (LLMs) have demonstrated the impact of prompting strategies and fine-tuning techniques on their reasoning capabilities. Yet, their effectiveness on clinical natural language inference (NLI) remains underexplored. This study presents the first controlled evaluation of how prompt structure and efficient fine-tuning jointly shape model performance in clinical NLI. We inspect four classes of prompting strategies to elicit reasoning in LLMs at different levels of abstraction, and evaluate their impact on a range of clinically motivated reasoning types. For each prompting strategy, we construct high-quality demonstrations using a frontier model to distil multi-step reasoning capabilities into smaller models (4B parameters) via Low-Rank Adaptation (LoRA). Across different language models fine-tuned on the NLI4CT benchmark, we found that prompt type alone accounts for up to 44% of the variance in macro-F1. Moreover, LoRA fine-tuning yields consistent gains of +8 to 12 F1, raises output alignment above 97%, and narrows the performance gap to GPT-4o-mini to within 7.1%. Additional experiments on reasoning generalisation reveal that LoRA improves performance in 75% of the models on MedNLI and TREC Clinical Trials Track. Overall, these findings demonstrate that (i) prompt structure is a primary driver of clinical reasoning performance, (ii) compact models equipped with strong prompts and LoRA can rival frontier-scale systems, and (iii) reasoning-type-aware evaluation is essential to uncover prompt-induced trade-offs. Our results highlight the promise of combining prompt design and lightweight adaptation for more efficient and trustworthy clinical NLP systems, providing insights on the strengths and limitations of widely adopted prompting and parameter-efficient techniques in highly specialised domains. 

**Abstract (ZH)**: 近期关于大规模语言模型（LLMs）的研究表明，提示策略和微调技术对模型推理能力有重大影响。然而，它们在临床自然语言推理（NLI）中的有效性仍待进一步探索。本研究首次系统评估了不同提示结构和高效微调方法如何共同影响临床NLI模型的性能。我们检查了四种不同抽象层次的提示策略，以激发LLMs的推理，并评估这些策略对多种临床动机推理类型的影响。对于每种提示策略，我们使用前沿模型构建高质量的演示，通过低秩适应（LoRA）将多步推理能力提炼到小模型（4B参数）中。在针对NLI4CT基准进行微调的不同语言模型上，我们发现提示类型自身可以解释高达44%的宏观F1值的变化。此外，LoRA微调提供了+8到12 F1的稳定增益，将输出对齐率提高到97%以上，并将性能差距缩小到GPT-4o-mini以内，仅差7.1%。额外的推理泛化实验结果显示，LoRA在MedNLI和TREC临床试验跟踪任务中提高了75%模型的性能。总体而言，这些发现表明：（i）提示结构是临床推理性能的主要驱动因素；（ii）配备强力提示和LoRA的紧凑模型可以与前沿系统媲美；（iii）具有推理类型的评估对于揭示提示引发的权衡至关重要。我们的结果强调了结合提示设计和轻量级适应以构建更高效和可信赖的临床NLP系统的潜力，提供了广泛采用的提示和参数高效技术在高度专业化领域中的优势和局限性见解。 

---
# Pedestrian Intention Prediction via Vision-Language Foundation Models 

**Title (ZH)**: 基于视觉-语言基础模型的行人意图预测 

**Authors**: Mohsen Azarmi, Mahdi Rezaei, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04141)  

**Abstract**: Prediction of pedestrian crossing intention is a critical function in autonomous vehicles. Conventional vision-based methods of crossing intention prediction often struggle with generalizability, context understanding, and causal reasoning. This study explores the potential of vision-language foundation models (VLFMs) for predicting pedestrian crossing intentions by integrating multimodal data through hierarchical prompt templates. The methodology incorporates contextual information, including visual frames, physical cues observations, and ego-vehicle dynamics, into systematically refined prompts to guide VLFMs effectively in intention prediction. Experiments were conducted on three common datasets-JAAD, PIE, and FU-PIP. Results demonstrate that incorporating vehicle speed, its variations over time, and time-conscious prompts significantly enhances the prediction accuracy up to 19.8%. Additionally, optimised prompts generated via an automatic prompt engineering framework yielded 12.5% further accuracy gains. These findings highlight the superior performance of VLFMs compared to conventional vision-based models, offering enhanced generalisation and contextual understanding for autonomous driving applications. 

**Abstract (ZH)**: 基于视觉-语言基础模型的多模态人行横道穿越意图预测 

---
# Driver-Net: Multi-Camera Fusion for Assessing Driver Take-Over Readiness in Automated Vehicles 

**Title (ZH)**: Driver-Net: 多摄像头融合评估自动驾驶车辆驾驶员接管准备状态 

**Authors**: Mahdi Rezaei, Mohsen Azarmi  

**Link**: [PDF](https://arxiv.org/pdf/2507.04139)  

**Abstract**: Ensuring safe transition of control in automated vehicles requires an accurate and timely assessment of driver readiness. This paper introduces Driver-Net, a novel deep learning framework that fuses multi-camera inputs to estimate driver take-over readiness. Unlike conventional vision-based driver monitoring systems that focus on head pose or eye gaze, Driver-Net captures synchronised visual cues from the driver's head, hands, and body posture through a triple-camera setup. The model integrates spatio-temporal data using a dual-path architecture, comprising a Context Block and a Feature Block, followed by a cross-modal fusion strategy to enhance prediction accuracy. Evaluated on a diverse dataset collected from the University of Leeds Driving Simulator, the proposed method achieves an accuracy of up to 95.8% in driver readiness classification. This performance significantly enhances existing approaches and highlights the importance of multimodal and multi-view fusion. As a real-time, non-intrusive solution, Driver-Net contributes meaningfully to the development of safer and more reliable automated vehicles and aligns with new regulatory mandates and upcoming safety standards. 

**Abstract (ZH)**: 确保自动驾驶车辆控制权过渡安全需要准确及时地评估驾驶员准备状态。本文引入了Driver-Net，一种新颖的深度学习框架，通过多摄像头输入融合来估计驾驶员接管准备状态。与传统的基于视觉的驾驶员监测系统侧重头姿或眼动不同，Driver-Net 通过三摄像头设置捕获驾驶员头部、手部和身体姿态的同步视觉线索。该模型使用包含上下文块和特征块的双重路径架构，并通过跨模态融合策略提高预测准确性。在利兹大学驾驶模拟器收集的多样数据集上评估，所提出的方法在驾驶员准备状态分类中的准确率达到95.8%。此性能显著提高了现有方法的效果，并强调了多模态和多视图融合的重要性。作为一种实时的非侵入性解决方案，Driver-Net 对更安全和更可靠的自动驾驶车辆的发展做出了重要贡献，并符合新的监管要求和即将出台的安全标准。 

---
# Towards Accurate and Efficient 3D Object Detection for Autonomous Driving: A Mixture of Experts Computing System on Edge 

**Title (ZH)**: 面向自动驾驶的准确高效3D物体检测：边缘计算环境下的混合专家系统 

**Authors**: Linshen Liu, Boyan Su, Junyue Jiang, Guanlin Wu, Cong Guo, Ceyu Xu, Hao Frank Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04123)  

**Abstract**: This paper presents Edge-based Mixture of Experts (MoE) Collaborative Computing (EMC2), an optimal computing system designed for autonomous vehicles (AVs) that simultaneously achieves low-latency and high-accuracy 3D object detection. Unlike conventional approaches, EMC2 incorporates a scenario-aware MoE architecture specifically optimized for edge platforms. By effectively fusing LiDAR and camera data, the system leverages the complementary strengths of sparse 3D point clouds and dense 2D images to generate robust multimodal representations. To enable this, EMC2 employs an adaptive multimodal data bridge that performs multi-scale preprocessing on sensor inputs, followed by a scenario-aware routing mechanism that dynamically dispatches features to dedicated expert models based on object visibility and distance. In addition, EMC2 integrates joint hardware-software optimizations, including hardware resource utilization optimization and computational graph simplification, to ensure efficient and real-time inference on resource-constrained edge devices. Experiments on open-source benchmarks clearly show the EMC2 advancements as a end-to-end system. On the KITTI dataset, it achieves an average accuracy improvement of 3.58% and a 159.06% inference speedup compared to 15 baseline methods on Jetson platforms, with similar performance gains on the nuScenes dataset, highlighting its capability to advance reliable, real-time 3D object detection tasks for AVs. 

**Abstract (ZH)**: 基于边缘的专家混合协作计算（EMC2）：面向自主车辆的低延迟高精度三维物体检测优化系统 

---
# When Data-Free Knowledge Distillation Meets Non-Transferable Teacher: Escaping Out-of-Distribution Trap is All You Need 

**Title (ZH)**: 当无数据知识精炼遇到非迁移性教师时：摆脱分布外陷阱即为所需 

**Authors**: Ziming Hong, Runnan Chen, Zengmao Wang, Bo Han, Bo Du, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04119)  

**Abstract**: Data-free knowledge distillation (DFKD) transfers knowledge from a teacher to a student without access the real in-distribution (ID) data. Its common solution is to use a generator to synthesize fake data and use them as a substitute for real ID data. However, existing works typically assume teachers are trustworthy, leaving the robustness and security of DFKD from untrusted teachers largely unexplored. In this work, we conduct the first investigation into distilling non-transferable learning (NTL) teachers using DFKD, where the transferability from an ID domain to an out-of-distribution (OOD) domain is prohibited. We find that NTL teachers fool DFKD through divert the generator's attention from the useful ID knowledge to the misleading OOD knowledge. This hinders ID knowledge transfer but prioritizes OOD knowledge transfer. To mitigate this issue, we propose Adversarial Trap Escaping (ATEsc) to benefit DFKD by identifying and filtering out OOD-like synthetic samples. Specifically, inspired by the evidence that NTL teachers show stronger adversarial robustness on OOD samples than ID samples, we split synthetic samples into two groups according to their robustness. The fragile group is treated as ID-like data and used for normal knowledge distillation, while the robust group is seen as OOD-like data and utilized for forgetting OOD knowledge. Extensive experiments demonstrate the effectiveness of ATEsc for improving DFKD against NTL teachers. Code is released at this https URL. 

**Abstract (ZH)**: 基于DFKD的非转移性学习教师知识蒸馏探究 

---
# Addressing The Devastating Effects Of Single-Task Data Poisoning In Exemplar-Free Continual Learning 

**Title (ZH)**: 无示例限制的连续学习中单一任务数据中毒的破坏性影响应对策略 

**Authors**: Stanisław Pawlak, Bartłomiej Twardowski, Tomasz Trzciński, Joost van de Weijer  

**Link**: [PDF](https://arxiv.org/pdf/2507.04106)  

**Abstract**: Our research addresses the overlooked security concerns related to data poisoning in continual learning (CL). Data poisoning - the intentional manipulation of training data to affect the predictions of machine learning models - was recently shown to be a threat to CL training stability. While existing literature predominantly addresses scenario-dependent attacks, we propose to focus on a more simple and realistic single-task poison (STP) threats. In contrast to previously proposed poisoning settings, in STP adversaries lack knowledge and access to the model, as well as to both previous and future tasks. During an attack, they only have access to the current task within the data stream. Our study demonstrates that even within these stringent conditions, adversaries can compromise model performance using standard image corruptions. We show that STP attacks are able to strongly disrupt the whole continual training process: decreasing both the stability (its performance on past tasks) and plasticity (capacity to adapt to new tasks) of the algorithm. Finally, we propose a high-level defense framework for CL along with a poison task detection method based on task vectors. The code is available at this https URL . 

**Abstract (ZH)**: 我们的研究关注连续学习中数据投毒的未被重视的安全问题。数据投毒——通过故意操纵训练数据从而影响机器学习模型的预测——最近被证明是对连续学习训练稳定性的威胁。尽管现有文献主要关注场景依赖性攻击，我们建议关注一种更加简单且现实的单一任务投毒（STP）威胁。与以往提出的投毒设置不同，在STP攻击中，攻击者缺乏对模型以及此前和未来任务的知识和访问权限，仅在攻击期间可访问数据流中的当前任务。我们的研究证明，在这些严格的条件下，攻击者仍然可以通过标准图像污染手段损害模型性能。我们表明，STP攻击能够强烈扰乱整个连续学习过程：减少了算法在以往任务上的稳定性和对新任务的可塑性（适应能力）。最后，我们提出了一种针对连续学习的高层防御框架，并提出了基于任务向量的任务投毒检测方法。代码可在以下链接获取。 

---
# Hierarchical Testing with Rabbit Optimization for Industrial Cyber-Physical Systems 

**Title (ZH)**: 基于兔子优化的工业 cyber-物理系统分层测试 

**Authors**: Jinwei Hu, Zezhi Tang, Xin Jin, Benyuan Zhang, Yi Dong, Xiaowei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04100)  

**Abstract**: This paper presents HERO (Hierarchical Testing with Rabbit Optimization), a novel black-box adversarial testing framework for evaluating the robustness of deep learning-based Prognostics and Health Management systems in Industrial Cyber-Physical Systems. Leveraging Artificial Rabbit Optimization, HERO generates physically constrained adversarial examples that align with real-world data distributions via global and local perspective. Its generalizability ensures applicability across diverse ICPS scenarios. This study specifically focuses on the Proton Exchange Membrane Fuel Cell system, chosen for its highly dynamic operational conditions, complex degradation mechanisms, and increasing integration into ICPS as a sustainable and efficient energy solution. Experimental results highlight HERO's ability to uncover vulnerabilities in even state-of-the-art PHM models, underscoring the critical need for enhanced robustness in real-world applications. By addressing these challenges, HERO demonstrates its potential to advance more resilient PHM systems across a wide range of ICPS domains. 

**Abstract (ZH)**: HERO（层级测试结合兔优化算法）：工业 cyber-物理系统中基于深度学习的 prognostics 和健康管理系统稳健性评估的新型黑盒对抗性测试框架 

---
# Conversation Forests: The Key to Fine Tuning Large Language Models for Multi-Turn Medical Conversations is Branching 

**Title (ZH)**: 对话森林：多轮医疗对话大型语言模型微调的关键是分支结构 

**Authors**: Thomas Savage  

**Link**: [PDF](https://arxiv.org/pdf/2507.04099)  

**Abstract**: Fine-tuning methods such as Direct Preference Optimization (DPO) and Group Relative Policy Optimization (GRPO) have demonstrated success in training large language models (LLMs) for single-turn tasks. However, these methods fall short in multi-turn applications, such as diagnostic patient interviewing, where understanding how early conversational turns influence downstream completions and outcomes is essential. In medicine, a multi-turn perspective is critical for learning diagnostic schemas and better understanding conversation dynamics. To address this gap, I introduce Savage Conversation Forests (SCF), a reinforcement learning framework that leverages a branched conversation architecture to fine-tune LLMs for multi-turn dialogue. SCF generates multiple possible conversation continuations at each turn, enabling the model to learn how different early responses affect downstream interactions and diagnostic outcomes. In experiments simulating doctor-patient conversations, SCF with branching outperforms linear conversation architectures on diagnostic accuracy. I hypothesize that SCF's improvements stem from its ability to provide richer, interdependent training signals across conversation turns. These results suggest that a branched training architecture is an important strategy for fine tuning LLMs in complex multi-turn conversational tasks. 

**Abstract (ZH)**: Savage Conversation Forests：一种用于复杂多轮对话任务的强化学习微调框架 

---
# Human-centered AI with focus on Human-robot interaction (Book chapter) 

**Title (ZH)**: 以人为中心的AI——以人机交互为重点（书章节） 

**Authors**: Alireza Mortezapour, Giuliana Vitiello  

**Link**: [PDF](https://arxiv.org/pdf/2507.04095)  

**Abstract**: Modern social robots can be considered the descendants of steam engines from the First Industrial Revolution (IR 1.0) and industrial robotic arms from the Third Industrial Revolution (IR 3.0). As some time has passed since the introduction of these robots during the Fourth Industrial Revolution (IR 4.0), challenges and issues in their interaction with humans have emerged, leading researchers to conclude that, like any other AI-based technology, these robots must also be human-centered to meet the needs of their users. This chapter aims to introduce humans and their needs in interactions with robots, ranging from short-term, one-on-one interactions (micro-level) to long-term, macro-level needs at the societal scale. Building upon the principles of human-centered AI, this chapter presents, for the first time, a new framework of human needs called the Dual Pyramid. This framework encompasses a comprehensive list of human needs in robot interactions, from the most fundamental, robot effectiveness to macro level requirements, such as the collaboration with robots in achieving the United Nations 17 Sustainable Development Goals. 

**Abstract (ZH)**: 现代社交机器人可以被视为第一工业革命（IR 1.0）中的蒸汽发动机和第三工业革命（IR 3.0）中的工业机器人臂的后裔。随着第四工业革命（IR 4.0）中这些机器人的引入时间推移，它们与人类互动中出现了挑战和问题，促使研究人员认识到，就像其他任何基于人工智能的技术一样，这些机器人也必须以人类为中心，以满足用户的需求。本章旨在介绍人类及其在与机器人互动中的需求，从短期一对一互动（微观层面）到长期、宏观层面的需求（社会层面）。基于以人为本的人工智能原则，本章首次提出了一种新的需求框架，称为双棱锥框架。该框架涵盖了机器人互动中全面的人类需求清单，从最基本的有效性需求到宏观层面的要求，如与机器人合作实现联合国17项可持续发展目标。 

---
# MMMOS: Multi-domain Multi-axis Audio Quality Assessment 

**Title (ZH)**: 多领域多轴音频质量评估 

**Authors**: Yi-Cheng Lin, Jia-Hung Chen, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.04094)  

**Abstract**: Accurate audio quality estimation is essential for developing and evaluating audio generation, retrieval, and enhancement systems. Existing non-intrusive assessment models predict a single Mean Opinion Score (MOS) for speech, merging diverse perceptual factors and failing to generalize beyond speech. We propose MMMOS, a no-reference, multi-domain audio quality assessment system that estimates four orthogonal axes: Production Quality, Production Complexity, Content Enjoyment, and Content Usefulness across speech, music, and environmental sounds. MMMOS fuses frame-level embeddings from three pretrained encoders (WavLM, MuQ, and M2D) and evaluates three aggregation strategies with four loss functions. By ensembling the top eight models, MMMOS shows a 20-30% reduction in mean squared error and a 4-5% increase in Kendall's {\tau} versus baseline, gains first place in six of eight Production Complexity metrics, and ranks among the top three on 17 of 32 challenge metrics. 

**Abstract (ZH)**: 准确的音频质量估计对于开发和评估音频生成、检索和增强系统至关重要。现有的非侵入性评估模型为语音预测单一的意见分数（MOS），合并了多种感知因素，并且无法泛化到非语音领域。我们提出MMMOS，这是一种无参考的多领域音频质量评估系统，估计四个正交轴：生产质量、生产复杂度、内容愉悦性和内容实用性，覆盖语音、音乐和环境声音。MMMOS 结合了三个预训练编码器（WavLM、MuQ 和 M2D）的帧级嵌入，并评估了三种聚合策略和四种损失函数。通过集成前八名模型，MMMOS 在均方误差上减少了 20-30%，在肯德尔 τ 上提高了 4-5%，在六个生产复杂度指标中排名第一，在 32 项挑战指标中的 17 项中排名前三。 

---
# Accurate and Efficient World Modeling with Masked Latent Transformers 

**Title (ZH)**: 掩码潜在变换器实现高效准确的世界建模 

**Authors**: Maxime Burchi, Radu Timofte  

**Link**: [PDF](https://arxiv.org/pdf/2507.04075)  

**Abstract**: The Dreamer algorithm has recently obtained remarkable performance across diverse environment domains by training powerful agents with simulated trajectories. However, the compressed nature of its world model's latent space can result in the loss of crucial information, negatively affecting the agent's performance. Recent approaches, such as $\Delta$-IRIS and DIAMOND, address this limitation by training more accurate world models. However, these methods require training agents directly from pixels, which reduces training efficiency and prevents the agent from benefiting from the inner representations learned by the world model. In this work, we propose an alternative approach to world modeling that is both accurate and efficient. We introduce EMERALD (Efficient MaskEd latent tRAnsformer worLD model), a world model using a spatial latent state with MaskGIT predictions to generate accurate trajectories in latent space and improve the agent performance. On the Crafter benchmark, EMERALD achieves new state-of-the-art performance, becoming the first method to surpass human experts performance within 10M environment steps. Our method also succeeds to unlock all 22 Crafter achievements at least once during evaluation. 

**Abstract (ZH)**: EMERALD：高效掩码潜在空间转换世界模型 

---
# Beyond Independent Passages: Adaptive Passage Combination Retrieval for Retrieval Augmented Open-Domain Question Answering 

**Title (ZH)**: 超越独立段落：自适应段落组合检索在开放域检索增强问答中的应用 

**Authors**: Ting-Wen Ko, Jyun-Yu Jiang, Pu-Jen Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.04069)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external documents at inference time, enabling up-to-date knowledge access without costly retraining. However, conventional RAG methods retrieve passages independently, often leading to redundant, noisy, or insufficiently diverse context-particularly problematic - particularly problematic in noisy corpora and for multi-hop questions. To address this, we propose Adaptive Passage Combination Retrieval (AdaPCR), a novel framework for open-domain question answering with black-box LMs. AdaPCR explicitly models dependencies between passages by considering passage combinations as units for retrieval and reranking. It consists of a context-aware query reformulation using concatenated passages, and a reranking step trained with a predictive objective aligned with downstream answer likelihood. Crucially, AdaPCR adaptively selects the number of retrieved passages without additional stopping modules. Experiments across several QA benchmarks show that AdaPCR outperforms baselines, particularly in multi-hop reasoning, demonstrating the effectiveness of modeling inter-passage dependencies for improved retrieval. 

**Abstract (ZH)**: 检索增强生成（RAG）通过在推理时 Incorporate 外部文档来增强大型语言模型（LLMs），从而使模型能够在不昂贵地重新训练的情况下访问实时知识。然而，传统的 RAG 方法单独检索段落，通常会导致冗余、噪声或上下文不够多样化——特别是在嘈杂的语料库和多跳问题中更为成问题。为此，我们提出了一种新颖的开放式领域问答框架 Adaptive Passage Combination Retrieval (AdaPCR)，该框架使用黑盒语言模型进行基于段落组合的检索和重排序。AdaPCR 通过考虑段落组合作为检索和重排序的基本单元来显式建模段落间的依赖性。它包括一种基于分段组合的上下文感知查询重写，以及一个与下游答案似然性对齐的重排序训练步骤。 crucial 地，AdaPCR 能够根据需要自适应地选择检索到的段落数量，而无需附加的停止模块。在多个 QA 度量标准上的实验表明，AdaPCR 在多跳推理方面优于基线方法，证明了建模段落间依赖性以改进检索的有效性。 

---
# Stochastic Human Motion Prediction with Memory of Action Transition and Action Characteristic 

**Title (ZH)**: 基于动作转换记忆和动作特征的随机人体运动预测 

**Authors**: Jianwei Tang, Hong Yang, Tengyue Chen, Jian-Fang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04062)  

**Abstract**: Action-driven stochastic human motion prediction aims to generate future motion sequences of a pre-defined target action based on given past observed sequences performing non-target actions. This task primarily presents two challenges. Firstly, generating smooth transition motions is hard due to the varying transition speeds of different actions. Secondly, the action characteristic is difficult to be learned because of the similarity of some actions. These issues cause the predicted results to be unreasonable and inconsistent. As a result, we propose two memory banks, the Soft-transition Action Bank (STAB) and Action Characteristic Bank (ACB), to tackle the problems above. The STAB stores the action transition information. It is equipped with the novel soft searching approach, which encourages the model to focus on multiple possible action categories of observed motions. The ACB records action characteristic, which produces more prior information for predicting certain actions. To fuse the features retrieved from the two banks better, we further propose the Adaptive Attention Adjustment (AAA) strategy. Extensive experiments on four motion prediction datasets demonstrate that our approach consistently outperforms the previous state-of-the-art. The demo and code are available at this https URL. 

**Abstract (ZH)**: 基于动作驱动的随机人体运动预测旨在根据给定的执行非目标动作的过去观察序列，生成预定义目标动作的未来运动序列。该任务主要面临两个挑战。首先，由于不同动作的过渡速度不同，生成平滑的过渡动作困难。其次，由于某些动作的相似性，动作特征难以学习。这些问题导致预测结果不合理且不一致。因此，我们提出了两种记忆库，软过渡动作库（STAB）和动作特性库（ACB），以解决上述问题。STAB存储动作过渡信息，并配备了新颖的软搜索方法，促使模型关注观察动作的多种可能动作类别。ACB记录动作特性，为预测特定动作提供更多的先验信息。为进一步更好地融合来自两个库的特征，我们还提出了自适应注意力调整（AAA）策略。在四个运动预测数据集上的广泛实验表明，我们的方法始终优于之前的最先进的方法。演示和代码可在以下网址获得。 

---
# Temporal Continual Learning with Prior Compensation for Human Motion Prediction 

**Title (ZH)**: 具有先验补偿的时空连续学习人体运动预测 

**Authors**: Jianwei Tang, Jiangxin Sun, Xiaotong Lin, Lifang Zhang, Wei-Shi Zheng, Jian-Fang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04060)  

**Abstract**: Human Motion Prediction (HMP) aims to predict future poses at different moments according to past motion sequences. Previous approaches have treated the prediction of various moments equally, resulting in two main limitations: the learning of short-term predictions is hindered by the focus on long-term predictions, and the incorporation of prior information from past predictions into subsequent predictions is limited. In this paper, we introduce a novel multi-stage training framework called Temporal Continual Learning (TCL) to address the above challenges. To better preserve prior information, we introduce the Prior Compensation Factor (PCF). We incorporate it into the model training to compensate for the lost prior information. Furthermore, we derive a more reasonable optimization objective through theoretical derivation. It is important to note that our TCL framework can be easily integrated with different HMP backbone models and adapted to various datasets and applications. Extensive experiments on four HMP benchmark datasets demonstrate the effectiveness and flexibility of TCL. The code is available at this https URL. 

**Abstract (ZH)**: 人类动作预测（HMP）旨在根据过去的动作序列预测不同时刻的未来姿态。以往的方法将不同时刻的预测视为等价的，导致了两个主要限制：短期预测的学习受到了长期预测重点的阻碍，先前预测中的先验信息向后续预测的融入有限。本文提出了一种新的多阶段训练框架——Temporal Continual Learning (TCL)，以应对上述挑战。为了更好地保留先验信息，引入了先验补偿因子（PCF），将其融入模型训练以补偿丢失的先验信息。此外，通过理论推导获得了更合理的优化目标。值得注意的是，我们的TCL框架可以轻松集成到不同的HMP骨干模型中，并适应各种数据集和应用场景。在四个HMP基准数据集上的广泛实验演示了TCL的有效性和灵活性。代码已开源。 

---
# Attributing Data for Sharpness-Aware Minimization 

**Title (ZH)**: Sharpness-Aware Minimization的数据贡献分析 

**Authors**: Chenyang Ren, Yifan Jia, Huanyi Xie, Zhaobin Xu, Tianxing Wei, Liangyu Wang, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04059)  

**Abstract**: Sharpness-aware Minimization (SAM) improves generalization in large-scale model training by linking loss landscape geometry to generalization. However, challenges such as mislabeled noisy data and privacy concerns have emerged as significant issues. Data attribution, which identifies the contributions of specific training samples, offers a promising solution. However, directly rendering existing data influence evaluation tools such as influence functions (IF) to SAM will be inapplicable or inaccurate as SAM utilizes an inner loop to find model perturbations that maximize loss, which the outer loop then minimizes, resulting in a doubled computational structure. Additionally, this bilevel structure complicates the modeling of data influence on the parameters. In this paper, based on the IF, we develop two innovative data valuation methods for SAM, each offering unique benefits in different scenarios: the Hessian-based IF and the Gradient Trajectory-based IF. The first one provides a comprehensive estimation of data influence using a closed-form measure that relies only on the trained model weights. In contrast, the other IF for SAM utilizes gradient trajectory information during training for more accurate and efficient data assessment. Extensive experiments demonstrate their effectiveness in data evaluation and parameter tuning, with applications in identifying mislabeled data, model editing, and enhancing interpretability. 

**Abstract (ZH)**: Sharpness-aware Minimization (SAM)通过将损失景观点几何与泛化能力联系起来从而在大规模模型训练中提高泛化能力，但mislabelled噪声数据和隐私问题等挑战已变得尤为突出。数据归属，即识别特定训练样本的贡献，提供了一个有前途的解决方案。然而，将现有的数据影响评估工具如影响函数（IF）直接应用于SAM是不适用或不准确的，因为SAM利用内部循环寻找使损失最大化的模型扰动，外部循环再最小化这些扰动，这导致了两层的计算结构，同时也增加了数据分析对参数影响建模的复杂性。基于IF，本文开发了两种创新的数据估值方法用于SAM，每种方法在不同的场景下各有独特优势：Hessian基于的IF和梯度轨迹基于的IF。前者通过仅依赖训练后的模型权重的封闭形式度量提供全面的数据影响估计；后者则利用训练期间的梯度轨迹信息进行更准确和高效的数据评估。广泛实验表明，这两种方法在数据评估和参数调优中具有有效性，并应用于识别错误标签数据、模型编辑和增强可解释性。 

---
# Rethinking and Exploring String-Based Malware Family Classification in the Era of LLMs and RAG 

**Title (ZH)**: 重思和探索大规模语言模型和 RETRIEVE-THEN-GENERATE 架构背景下基于字符串的恶意软件家族分类 

**Authors**: Yufan Chen, Daoyuan Wu, Juantao Zhong, Zicheng Zhang, Debin Gao, Shuai Wang, Yingjiu Li, Ning Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04055)  

**Abstract**: Malware Family Classification (MFC) aims to identify the fine-grained family (e.g., GuLoader or BitRAT) to which a potential malware sample belongs, in contrast to malware detection or sample classification that predicts only an Yes/No. Accurate family identification can greatly facilitate automated sample labeling and understanding on crowdsourced malware analysis platforms such as VirusTotal and MalwareBazaar, which generate vast amounts of data daily. In this paper, we explore and assess the feasibility of using traditional binary string features for MFC in the new era of large language models (LLMs) and Retrieval-Augmented Generation (RAG). Specifically, we investigate how Family-Specific String (FSS) features could be utilized in a manner similar to RAG to facilitate MFC. To this end, we develop a curated evaluation framework covering 4,347 samples from 67 malware families, extract and analyze over 25 million strings, and conduct detailed ablation studies to assess the impact of different design choices in four major modules. 

**Abstract (ZH)**: 恶意软件家族分类（MFC）旨在识别一个潜在的恶意软件样本属于哪个细粒度家族（例如GuLoader或BitRAT），而不仅仅是预测Yes/No。准确的家族识别可以大大促进在VirusTotal和MalwareBazaar等众包恶意软件分析平台上自动样本标签化和理解，这些平台每天生成大量数据。在本文中，我们探讨并评估了在大规模语言模型（LLMs）和检索增强生成（RAG）的新时代，使用传统二进制字符串特征进行MFC的可能性。具体而言，我们研究了如何利用家族特定字符串（FSS）特征，使其类似RAG的方式促进MFC。为此，我们开发了一个包含来自67个恶意软件家族的4,347个样本的定制评估框架，提取并分析了超过2500万个字符串，并进行了详细的消融研究以评估四大模块中不同设计选择的影响。 

---
# TopoMAS: Large Language Model Driven Topological Materials Multiagent System 

**Title (ZH)**: TopoMAS：由大规模语言模型驱动的拓扑材料多智能体系统 

**Authors**: Baohua Zhang, Xin Li, Huangchao Xu, Zhong Jin, Quansheng Wu, Ce Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.04053)  

**Abstract**: Topological materials occupy a frontier in condensed-matter physics thanks to their remarkable electronic and quantum properties, yet their cross-scale design remains bottlenecked by inefficient discovery workflows. Here, we introduce TopoMAS (Topological materials Multi-Agent System), an interactive human-AI framework that seamlessly orchestrates the entire materials-discovery pipeline: from user-defined queries and multi-source data retrieval, through theoretical inference and crystal-structure generation, to first-principles validation. Crucially, TopoMAS closes the loop by autonomously integrating computational outcomes into a dynamic knowledge graph, enabling continuous knowledge refinement. In collaboration with human experts, it has already guided the identification of novel topological phases SrSbO3, confirmed by first-principles calculations. Comprehensive benchmarks demonstrate robust adaptability across base Large Language Model, with the lightweight Qwen2.5-72B model achieving 94.55% accuracy while consuming only 74.3-78.4% of tokens required by Qwen3-235B and 83.0% of DeepSeek-V3's usage--delivering responses twice as fast as Qwen3-235B. This efficiency establishes TopoMAS as an accelerator for computation-driven discovery pipelines. By harmonizing rational agent orchestration with a self-evolving knowledge graph, our framework not only delivers immediate advances in topological materials but also establishes a transferable, extensible paradigm for materials-science domain. 

**Abstract (ZH)**: 拓扑材料因其独特的电子和量子性质，在凝聚态物理学中占据前沿地位，然而其跨尺度设计仍受限于低效的发现工作流程。在这里，我们引入了拓扑材料多智能体系统（TopoMAS），一个交互式的人工智能框架，无缝 orchestrates 整个材料发现流程：从用户定义的查询和多源数据检索，到理论推断和晶体结构生成，最后进行第一性原理验证。至关重要的是，TopoMAS 自动将计算结果整合到动态知识图谱中，形成持续的知识精炼。与人类专家合作，它已指导识别出新型拓扑相 SrSbO3，由第一性原理计算确认。全面的基准测试表明，该框架在基本大型语言模型中表现出高度的适应性，采用轻量级的 Qwen2.5-72B 模型实现了 94.55% 的准确性，仅消耗 Qwen3-235B 和 DeepSeek-V3 所需令牌的 74.3-78.4% 和 83.0%——速度比 Qwen3-235B 快两倍。这种效率使 TopoMAS 成为驱动计算导向的发现管道的加速器。通过合理智能体协调与自我进化知识图谱的和谐统一，我们的框架不仅在拓扑材料领域立即实现了进展，还为材料科学领域建立了可转移和可扩展的范式。 

---
# Predictive Modeling of Effluent Temperature in SAT Systems Using Ambient Meteorological Data: Implications for Infiltration Management 

**Title (ZH)**: 基于环境气象数据的SAT系统排放温度预测模型：渗漏管理的 implications 

**Authors**: Roy Elkayam  

**Link**: [PDF](https://arxiv.org/pdf/2507.04050)  

**Abstract**: Accurate prediction of effluent temperature in recharge basins is essential for optimizing the Soil Aquifer Treatment (SAT) process, as temperature directly influences water viscosity and infiltration rates. This study develops and evaluates predictive models for effluent temperature in the upper recharge layer of a Shafdan SAT system recharge basin using ambient meteorological data. Multiple linear regression (MLR), neural networks (NN), and random forests (RF) were tested for their predictive accuracy and interpretability. The MLR model, preferred for its operational simplicity and robust performance, achieved high predictive accuracy (R2 = 0.86-0.87) and was used to estimate effluent temperatures over a 10-year period. Results highlight pronounced seasonal temperature cycles and the importance of topsoil temperature in governing the thermal profile of the infiltrating effluent. The study provides practical equations for real-time monitoring and long-term planning of SAT operations. 

**Abstract (ZH)**: 准确预测砂石渗滤系统 recharge basin 中溢流水温对于优化土壤渗滤处理（SAT）过程至关重要，因为水温直接影响水的黏度和渗透率。本研究使用环境气象数据开发并评估了用于预测 Shafdan SAT 系统 recharge basin 上层渗滤层溢流水温的预测模型。测试了多元线性回归（MLR）、神经网络（NN）和随机森林（RF）模型的预测准确性和可解释性。MLR 模型因其操作简单和稳健性而被首选，实现了高预测准确度（R2 = 0.86-0.87），并用于估算10年内的溢流水温。研究结果强调了明显的季节性温度周期以及表层土温对入渗溢流水温度热特征的控制作用。本研究提供了实际方程，用于 SAT 运行的实时监控和长期规划。 

---
# Evaluating the Effectiveness of Large Language Models in Solving Simple Programming Tasks: A User-Centered Study 

**Title (ZH)**: 评估大型语言模型解决简单编程任务的有效性：一项以用户为中心的研究 

**Authors**: Kai Deng  

**Link**: [PDF](https://arxiv.org/pdf/2507.04043)  

**Abstract**: As large language models (LLMs) become more common in educational tools and programming environments, questions arise about how these systems should interact with users. This study investigates how different interaction styles with ChatGPT-4o (passive, proactive, and collaborative) affect user performance on simple programming tasks. I conducted a within-subjects experiment where fifteen high school students participated, completing three problems under three distinct versions of the model. Each version was designed to represent a specific style of AI support: responding only when asked, offering suggestions automatically, or engaging the user in back-and-forth this http URL analysis revealed that the collaborative interaction style significantly improved task completion time compared to the passive and proactive conditions. Participants also reported higher satisfaction and perceived helpfulness when working with the collaborative version. These findings suggest that the way an LLM communicates, how it guides, prompts, and responds, can meaningfully impact learning and performance. This research highlights the importance of designing LLMs that go beyond functional correctness to support more interactive, adaptive, and user-centered experiences, especially for novice programmers. 

**Abstract (ZH)**: 大型语言模型（LLMs）在教育工具和编程环境中变得更为常见后，关于这些系统应如何与用户交互的问题引起了人们的关注。本研究探讨了不同的ChatGPT-4o交互风格（被动、主动和协作）对用户完成简单编程任务性能的影响。我进行了一项被试内实验，十五名高中生参与试验，在三种不同的模型版本下完成了三项问题。每种版本旨在代表特定的AI支持风格：仅在被询问时响应、自动提供建议或与用户进行互动。分析表明，与被动和主动条件相比，协作交互风格显著减少了任务完成时间。参与者还认为与协作版本合作时的满意度和感知到的帮助更大。这些发现表明，LLM的通信方式、引导、提示和响应的方式对学习和表现有实质性的影响。本研究强调了设计超越功能性正确性、支持更互动、适应性和用户中心体验的LLM的重要性，尤其是在为初学者编程者服务方面。 

---
# T-SYNTH: A Knowledge-Based Dataset of Synthetic Breast Images 

**Title (ZH)**: T-SYNTH：基于知识的合成乳腺图像数据集 

**Authors**: Christopher Wiedeman, Anastasiia Sarmakeeva, Elena Sizikova, Daniil Filienko, Miguel Lago, Jana G. Delfino, Aldo Badano  

**Link**: [PDF](https://arxiv.org/pdf/2507.04038)  

**Abstract**: One of the key impediments for developing and assessing robust medical imaging algorithms is limited access to large-scale datasets with suitable annotations. Synthetic data generated with plausible physical and biological constraints may address some of these data limitations. We propose the use of physics simulations to generate synthetic images with pixel-level segmentation annotations, which are notoriously difficult to obtain. Specifically, we apply this approach to breast imaging analysis and release T-SYNTH, a large-scale open-source dataset of paired 2D digital mammography (DM) and 3D digital breast tomosynthesis (DBT) images. Our initial experimental results indicate that T-SYNTH images show promise for augmenting limited real patient datasets for detection tasks in DM and DBT. Our data and code are publicly available at this https URL. 

**Abstract (ZH)**: 生成具有像素级分割注释的合成医学影像数据以克服大規模标注数据受限的瓶颈：T-SYNTH数据集在乳腺影像分析中的应用 

---
# Nunchi-Bench: Benchmarking Language Models on Cultural Reasoning with a Focus on Korean Superstition 

**Title (ZH)**: Nunchi-Bench：基于韩国 superstition 的文化推理语言模型基准测试 

**Authors**: Kyuhee Kim, Sangah Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.04014)  

**Abstract**: As large language models (LLMs) become key advisors in various domains, their cultural sensitivity and reasoning skills are crucial in multicultural environments. We introduce Nunchi-Bench, a benchmark designed to evaluate LLMs' cultural understanding, with a focus on Korean superstitions. The benchmark consists of 247 questions spanning 31 topics, assessing factual knowledge, culturally appropriate advice, and situational interpretation. We evaluate multilingual LLMs in both Korean and English to analyze their ability to reason about Korean cultural contexts and how language variations affect performance. To systematically assess cultural reasoning, we propose a novel evaluation strategy with customized scoring metrics that capture the extent to which models recognize cultural nuances and respond appropriately. Our findings highlight significant challenges in LLMs' cultural reasoning. While models generally recognize factual information, they struggle to apply it in practical scenarios. Furthermore, explicit cultural framing enhances performance more effectively than relying solely on the language of the prompt. To support further research, we publicly release Nunchi-Bench alongside a leaderboard. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在各个领域成为关键顾问，它们在多文化环境中对文化的敏感性和推理能力至关重要。我们介绍了Nunchi-Bench，这是一个旨在评估LLMs文化理解的基准，尤其关注韩国迷信。该基准包括247个问题，涵盖31个主题，评估事实知识、文化适宜建议以及情境解读。我们用韩语和英语评估多语言LLMs，以分析其理解韩国文化背景的能力以及语言差异如何影响性能。为了系统评估文化推理能力，我们提出了一个新的评估策略，包含定制的评分标准，可以捕捉模型识别文化细微差别并作出适当反应的程度。我们的研究发现突显了LLMs在文化推理方面的重要挑战。尽管模型通常能识别事实信息，但在实际场景中应用这些信息却颇具困难。此外，明确的文化背景设定比仅仅依赖提示语言能更有效地提高性能。为了支持进一步研究，我们公开发布了Nunchi-Bench及其排行榜。 

---
# Leveraging Multimodal Data and Side Users for Diffusion Cross-Domain Recommendation 

**Title (ZH)**: 利用多模态数据和侧用户进行扩散跨域推荐 

**Authors**: Fan Zhang, Jinpeng Chen, Huan Li, Senzhang Wang, Yuan Cao, Kaimin Wei, JianXiang He, Feifei Kou, Jinqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04000)  

**Abstract**: Cross-domain recommendation (CDR) aims to address the persistent cold-start problem in Recommender Systems. Current CDR research concentrates on transferring cold-start users' information from the auxiliary domain to the target domain. However, these systems face two main issues: the underutilization of multimodal data, which hinders effective cross-domain alignment, and the neglect of side users who interact solely within the target domain, leading to inadequate learning of the target domain's vector space distribution. To address these issues, we propose a model leveraging Multimodal data and Side users for diffusion Cross-domain recommendation (MuSiC). We first employ a multimodal large language model to extract item multimodal features and leverage a large language model to uncover user features using prompt learning without fine-tuning. Secondly, we propose the cross-domain diffusion module to learn the generation of feature vectors in the target domain. This approach involves learning feature distribution from side users and understanding the patterns in cross-domain transformation through overlapping users. Subsequently, the trained diffusion module is used to generate feature vectors for cold-start users in the target domain, enabling the completion of cross-domain recommendation tasks. Finally, our experimental evaluation of the Amazon dataset confirms that MuSiC achieves state-of-the-art performance, significantly outperforming all selected baselines. Our code is available: this https URL. 

**Abstract (ZH)**: 跨域推荐（CDR）旨在解决推荐系统中的持续冷启动问题。当前的跨域推荐研究主要集中在从辅助域向目标域转移冷启动用户的信息。然而，这些系统面临两个主要问题：多模态数据的利用不足，这阻碍了有效的跨域对齐，以及忽视仅在目标域内交互的旁用户，导致对目标域向量空间分布的学习不足。为了解决这些问题，我们提出了一种利用多模态数据和旁用户的扩散跨域推荐模型（MuSiC）。我们首先使用多模态大型语言模型提取项目多模态特征，并利用大型语言模型通过提示学习发现用户特征，无需微调。其次，我们提出跨域扩散模块来学习目标域中特征向量的生成。该方法通过交错用户学习特征分布，并通过跨域转换理解模式。随后，训练好的扩散模块用于生成目标域冷启动用户的特征向量，从而完成跨域推荐任务。最后，我们在亚马逊数据集上的实验评估证实MuSiC达到了最先进的性能，显著优于所有选定的基线。相关代码可在以下链接获取：this https URL。 

---
# Real-TabPFN: Improving Tabular Foundation Models via Continued Pre-training With Real-World Data 

**Title (ZH)**: 实数据继续预训练提升表单基础模型：Real-TabPFN研究 

**Authors**: Anurag Garg, Muhammad Ali, Noah Hollmann, Lennart Purucker, Samuel Müller, Frank Hutter  

**Link**: [PDF](https://arxiv.org/pdf/2507.03971)  

**Abstract**: Foundation models for tabular data, like TabPFN, achieve strong performance on small datasets when pre-trained solely on synthetic data. We show that this performance can be significantly boosted by a targeted continued pre-training phase. Specifically, we demonstrate that leveraging a small, curated collection of large, real-world datasets for continued pre-training yields superior downstream predictive accuracy compared to using broader, potentially noisier corpora like CommonCrawl or GitTables. Our resulting model, Real-TabPFN, achieves substantial performance gains on 29 datasets from the OpenML AutoML Benchmark. 

**Abstract (ZH)**: 基于合成数据预训练的表格式数据基础模型，如TabPFN，在继续使用精心策划的大规模真实世界数据集进行微调后，能在小数据集上实现显著的性能提升。我们的Real-TabPFN模型在OpenML AutoML基准测试的29个数据集上取得了实质性的性能提升。 

---
# A Comparative Study of Specialized LLMs as Dense Retrievers 

**Title (ZH)**: 专业型大语言模型作为密集检索器的比较研究 

**Authors**: Hengran Zhang, Keping Bi, Jiafeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.03958)  

**Abstract**: While large language models (LLMs) are increasingly deployed as dense retrievers, the impact of their domain-specific specialization on retrieval effectiveness remains underexplored. This investigation systematically examines how task-specific adaptations in LLMs influence their retrieval capabilities, an essential step toward developing unified retrievers capable of handling text, code, images, and multimodal content. We conduct extensive experiments with eight Qwen2.5 7B LLMs, including base, instruction-tuned, code/math-specialized, long reasoning, and vision-language models across zero-shot retrieval settings and the supervised setting. For the zero-shot retrieval settings, we consider text retrieval from the BEIR benchmark and code retrieval from the CoIR benchmark. Further, to evaluate supervised performance, all LLMs are fine-tuned on the MS MARCO dataset. We find that mathematical specialization and the long reasoning capability cause consistent degradation in three settings, indicating conflicts between mathematical reasoning and semantic matching. The vision-language model and code-specialized LLMs demonstrate superior zero-shot performance compared to other LLMs, even surpassing BM25 on the code retrieval task, and maintain comparable performance to base LLMs in supervised settings. These findings suggest promising directions for the unified retrieval task leveraging cross-domain and cross-modal fusion. 

**Abstract (ZH)**: 大规模语言模型（LLMs）作为密集检索器的应用日益增多，但它们的专业化程度对其检索效果的影响尚未被充分探索。本研究系统地考察了任务特定适应性如何影响LLMs的检索能力，这是开发能够处理文本、代码、图像和多模态内容的统一检索器的重要一步。我们使用八个Qwen2.5 7B LLMs进行了广泛的实验，包括基础模型、指令调优模型、代码/数学专业化模型、长推理能力和视觉语言模型，在零样本检索和监督设置下进行考察。对于零样本检索设置，我们考虑了BEIR基准的文本检索和CoIR基准的代码检索。此外，为了评估监督性能，所有模型都在MS MARCO数据集上进行了微调。我们发现数学专业化和长推理能力在三个设置中导致了一致的性能下降，表明数学推理与语义匹配之间存在冲突。视觉语言模型和代码专业化模型在零样本性能上优于其他模型，甚至在代码检索任务中超过BM25，并在监督设置中保持与基础模型相当的性能。这些发现表明，在跨域和跨模态融合下，统一检索任务具有广阔的发展前景。 

---
# Evaluating Adversarial Protections for Diffusion Personalization: A Comprehensive Study 

**Title (ZH)**: 评估对抗保护在扩散个性化中的效果：一项全面研究 

**Authors**: Kai Ye, Tianyi Chen, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03953)  

**Abstract**: With the increasing adoption of diffusion models for image generation and personalization, concerns regarding privacy breaches and content misuse have become more pressing. In this study, we conduct a comprehensive comparison of eight perturbation based protection methods: AdvDM, ASPL, FSGM, MetaCloak, Mist, PhotoGuard, SDS, and SimAC--across both portrait and artwork domains. These methods are evaluated under varying perturbation budgets, using a range of metrics to assess visual imperceptibility and protective efficacy. Our results offer practical guidance for method selection. Code is available at: this https URL. 

**Abstract (ZH)**: 随着扩散模型在图像生成和个人化应用中的日益普及，隐私泄露和内容滥用的问题变得更为紧迫。本研究对八种基于扰动的保护方法——AdvDM、ASPL、FSGM、MetaCloak、Mist、PhotoGuard、SDS和SimAC——在肖像和艺术作品领域进行了全面比较。这些方法在不同的扰动预算下进行评估，并使用多种指标来评估视觉不可感知性和保护效力。研究结果提供了方法选择的实际指导。代码可在以下链接获取：this https URL。 

---
# Optimizing Age of Trust and Throughput in Multi-Hop UAV-Aided IoT Networks 

**Title (ZH)**: 优化多跳UAV辅助物联网网络中的信任年龄和吞吐量 

**Authors**: Yizhou Luo, Kwan-Wu Chin, Ruyi Guan, Xi Xiao, Caimeng Wang, Jingyin Feng, Tengjiao He  

**Link**: [PDF](https://arxiv.org/pdf/2507.03950)  

**Abstract**: Devices operating in Internet of Things (IoT) networks may be deployed across vast geographical areas and interconnected via multi-hop communications. Further, they may be unguarded. This makes them vulnerable to attacks and motivates operators to check on devices frequently. To this end, we propose and study an Unmanned Aerial Vehicle (UAV)-aided attestation framework for use in IoT networks with a charging station powered by solar. A key challenge is optimizing the trajectory of the UAV to ensure it attests as many devices as possible. A trade-off here is that devices being checked by the UAV are offline, which affects the amount of data delivered to a gateway. Another challenge is that the charging station experiences time-varying energy arrivals, which in turn affect the flight duration and charging schedule of the UAV. To address these challenges, we employ a Deep Reinforcement Learning (DRL) solution to optimize the UAV's charging schedule and the selection of devices to be attested during each flight. The simulation results show that our solution reduces the average age of trust by 88% and throughput loss due to attestation by 30%. 

**Abstract (ZH)**: 基于太阳能充电站的无人机辅助物联网网络验证框架及其优化研究 

---
# EdgeSRIE: A hybrid deep learning framework for real-time speckle reduction and image enhancement on portable ultrasound systems 

**Title (ZH)**: EdgeSRIE：便携超声系统中实时 speckle 减少和图像增强的混合深度学习框架 

**Authors**: Hyunwoo Cho, Jongsoo Lee, Jinbum Kang, Yangmo Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2507.03937)  

**Abstract**: Speckle patterns in ultrasound images often obscure anatomical details, leading to diagnostic uncertainty. Recently, various deep learning (DL)-based techniques have been introduced to effectively suppress speckle; however, their high computational costs pose challenges for low-resource devices, such as portable ultrasound systems. To address this issue, EdgeSRIE, which is a lightweight hybrid DL framework for real-time speckle reduction and image enhancement in portable ultrasound imaging, is introduced. The proposed framework consists of two main branches: an unsupervised despeckling branch, which is trained by minimizing a loss function between speckled images, and a deblurring branch, which restores blurred images to sharp images. For hardware implementation, the trained network is quantized to 8-bit integer precision and deployed on a low-resource system-on-chip (SoC) with limited power consumption. In the performance evaluation with phantom and in vivo analyses, EdgeSRIE achieved the highest contrast-to-noise ratio (CNR) and average gradient magnitude (AGM) compared with the other baselines (different 2-rule-based methods and other 4-DL-based methods). Furthermore, EdgeSRIE enabled real-time inference at over 60 frames per second while satisfying computational requirements (< 20K parameters) on actual portable ultrasound hardware. These results demonstrated the feasibility of EdgeSRIE for real-time, high-quality ultrasound imaging in resource-limited environments. 

**Abstract (ZH)**: 边缘SRIE：一种基于轻量级混合深度学习框架的便携超声成像实时去噪与图像增强方法 

---
# Learning Disentangled Stain and Structural Representations for Semi-Supervised Histopathology Segmentation 

**Title (ZH)**: 学习解耦染色和结构表示的半监督病理分割 

**Authors**: Ha-Hieu Pham, Nguyen Lan Vi Vu, Thanh-Huy Nguyen, Ulas Bagci, Min Xu, Trung-Nghia Le, Huy-Hieu Pham  

**Link**: [PDF](https://arxiv.org/pdf/2507.03923)  

**Abstract**: Accurate gland segmentation in histopathology images is essential for cancer diagnosis and prognosis. However, significant variability in Hematoxylin and Eosin (H&E) staining and tissue morphology, combined with limited annotated data, poses major challenges for automated segmentation. To address this, we propose Color-Structure Dual-Student (CSDS), a novel semi-supervised segmentation framework designed to learn disentangled representations of stain appearance and tissue structure. CSDS comprises two specialized student networks: one trained on stain-augmented inputs to model chromatic variation, and the other on structure-augmented inputs to capture morphological cues. A shared teacher network, updated via Exponential Moving Average (EMA), supervises both students through pseudo-labels. To further improve label reliability, we introduce stain-aware and structure-aware uncertainty estimation modules that adaptively modulate the contribution of each student during training. Experiments on the GlaS and CRAG datasets show that CSDS achieves state-of-the-art performance in low-label settings, with Dice score improvements of up to 1.2% on GlaS and 0.7% on CRAG at 5% labeled data, and 0.7% and 1.4% at 10%. Our code and pre-trained models are available at this https URL. 

**Abstract (ZH)**: 准确的腺体分割对于癌症诊断和预后至关重要。然而，H&E染色和组织形态的巨大变异性以及标注数据的限制，为自动分割带来了重大挑战。为此，我们提出了一种新的半监督分割框架Color-Structure Dual-Student (CSDS)，该框架旨在学习染色外观和组织结构的去混纠缠表征。CSDS 包含两个专门的学生网络：一个在增强染色输入上训练以建模色度变化，另一个在增强结构输入上训练以捕捉形态学线索。通过指数移动平均(EMA)更新的共享教师网络通过伪标签监督两个学生。为了进一步提高标签的可靠性，我们引入了染色感知和结构感知的不确定性估计模块，这些模块在训练过程中适当地调节每个学生贡献的权重。在GlaS和CRAG数据集上的实验表明，CSDS 在低标签设置中达到了最先进的性能，在5%和10%标记数据的情况下，GlaS的Dice分数分别提高了1.2%和0.7%，CRAG分别提高了0.7%和1.4%。我们的代码和预训练模型可在以下链接获取。 

---
# Transformer Model for Alzheimer's Disease Progression Prediction Using Longitudinal Visit Sequences 

**Title (ZH)**: 基于纵向访视序列的Transformer模型在阿尔茨海默病进展预测中的应用 

**Authors**: Mahdi Moghaddami, Clayton Schubring, Mohammad-Reza Siadat  

**Link**: [PDF](https://arxiv.org/pdf/2507.03899)  

**Abstract**: Alzheimer's disease (AD) is a neurodegenerative disorder with no known cure that affects tens of millions of people worldwide. Early detection of AD is critical for timely intervention to halt or slow the progression of the disease. In this study, we propose a Transformer model for predicting the stage of AD progression at a subject's next clinical visit using features from a sequence of visits extracted from the subject's visit history. We also rigorously compare our model to recurrent neural networks (RNNs) such as long short-term memory (LSTM), gated recurrent unit (GRU), and minimalRNN and assess their performances based on factors such as the length of prior visits and data imbalance. We test the importance of different feature categories and visit history, as well as compare the model to a newer Transformer-based model optimized for time series. Our model demonstrates strong predictive performance despite missing visits and missing features in available visits, particularly in identifying converter subjects -- individuals transitioning to more severe disease stages -- an area that has posed significant challenges in longitudinal prediction. The results highlight the model's potential in enhancing early diagnosis and patient outcomes. 

**Abstract (ZH)**: 阿尔茨海默病（AD）是一种无法治愈的神经退行性疾病，全球有数千万人受到其影响。早期检测AD对于及时干预以阻止或减缓疾病进展至关重要。在本研究中，我们提出了一种Transformer模型，用于预测患者下次临床访视时的AD进展阶段，该模型利用从访视历史中提取的访视序列特征。我们还严格比较了该模型与长短期记忆网络（LSTM）、门控循环单元（GRU）和最小RNN等循环神经网络（RNNs）的性能，并根据先前访视长度和数据不平衡等因素评估其表现。我们测试了不同特征类别和访视历史的重要性，并将该模型与一种针对时间序列优化的最新Transformer模型进行了比较。尽管存在缺失访视和可用访视中的特征缺失，我们的模型仍表现出强大的预测性能，特别是在识别转换者（即向更严重疾病阶段过渡的个体）方面表现出色，这为纵向预测带来了重大挑战。研究结果凸显了该模型在早期诊断和改善患者结果方面的潜力。 

---
# TayFCS: Towards Light Feature Combination Selection for Deep Recommender Systems 

**Title (ZH)**: TayFCS: 向量轻特征组合选择助力深度推荐系统 

**Authors**: Xianquan Wang, Zhaocheng Du, Jieming Zhu, Chuhan Wu, Qinglin Jia, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2507.03895)  

**Abstract**: Feature interaction modeling is crucial for deep recommendation models. A common and effective approach is to construct explicit feature combinations to enhance model performance. However, in practice, only a small fraction of these combinations are truly informative. Thus it is essential to select useful feature combinations to reduce noise and manage memory consumption. While feature selection methods have been extensively studied, they are typically limited to selecting individual features. Extending these methods for high-order feature combination selection presents a significant challenge due to the exponential growth in time complexity when evaluating feature combinations one by one. In this paper, we propose $\textbf{TayFCS}$, a lightweight feature combination selection method that significantly improves model performance. Specifically, we propose the Taylor Expansion Scorer (TayScorer) module for field-wise Taylor expansion on the base model. Instead of evaluating all potential feature combinations' importance by repeatedly running experiments with feature adding and removal, this scorer only needs to approximate the importance based on their sub-components' gradients. This can be simply computed with one backward pass based on a trained recommendation model. To further reduce information redundancy among feature combinations and their sub-components, we introduce Logistic Regression Elimination (LRE), which estimates the corresponding information gain based on the model prediction performance. Experimental results on three benchmark datasets validate both the effectiveness and efficiency of our approach. Furthermore, online A/B test results demonstrate its practical applicability and commercial value. 

**Abstract (ZH)**: 特征组合选择对于深度推荐模型至关重要。一种常见且有效的方法是构造显式的特征组合以提升模型性能。然而，在实践中，只有少量的组合真正具有信息性。因此，选择有用的特征组合以降低噪声和管理内存消耗是必要的。尽管特征选择方法已经被广泛研究，但它们通常仅限于选择单个特征。由于逐个评估特征组合的重要性的计算时间呈指数级增长，因此将这些方法扩展到高阶特征组合选择是一项重大挑战。在本文中，我们提出了一种轻量级的特征组合选择方法 $\textbf{TayFCS}$，显著提升了模型性能。具体地，我们提出了场-wise泰勒展开评分模块（TayScorer）来基于基模型进行场-wise泰勒展开。该评分器不需要通过反复运行添加和移除特征的实验来评估所有潜在的特征组合的重要性，而只需基于其子组件的梯度进行重要性的近似估计。这可以通过在训练好的推荐模型基础上进行一次反向传播计算得出。为了进一步减少特征组合及其子组件之间的信息冗余，我们引入了逻辑回归消除（LRE），该方法基于模型预测性能估计相应的信息增益。实验结果在三个基准数据集上验证了该方法的有效性和效率。此外，线上 A/B 测试结果展示了其实际适用性和商业价值。 

---
# Hierarchical Semantic-Visual Fusion of Visible and Near-infrared Images for Long-range Haze Removal 

**Title (ZH)**: 可见光和近红外图像分层语义-视觉融合在长距离消雾中的应用 

**Authors**: Yi Li, Xiaoxiong Wang, Jiawei Wang, Yi Chang, Kai Cao, Luxin Yan  

**Link**: [PDF](https://arxiv.org/pdf/2507.03893)  

**Abstract**: While image dehazing has advanced substantially in the past decade, most efforts have focused on short-range scenarios, leaving long-range haze removal under-explored. As distance increases, intensified scattering leads to severe haze and signal loss, making it impractical to recover distant details solely from visible images. Near-infrared, with superior fog penetration, offers critical complementary cues through multimodal fusion. However, existing methods focus on content integration while often neglecting haze embedded in visible images, leading to results with residual haze. In this work, we argue that the infrared and visible modalities not only provide complementary low-level visual features, but also share high-level semantic consistency. Motivated by this, we propose a Hierarchical Semantic-Visual Fusion (HSVF) framework, comprising a semantic stream to reconstruct haze-free scenes and a visual stream to incorporate structural details from the near-infrared modality. The semantic stream first acquires haze-robust semantic prediction by aligning modality-invariant intrinsic representations. Then the shared semantics act as strong priors to restore clear and high-contrast distant scenes under severe haze degradation. In parallel, the visual stream focuses on recovering lost structural details from near-infrared by fusing complementary cues from both visible and near-infrared images. Through the cooperation of dual streams, HSVF produces results that exhibit both high-contrast scenes and rich texture details. Moreover, we introduce a novel pixel-aligned visible-infrared haze dataset with semantic labels to facilitate benchmarking. Extensive experiments demonstrate the superiority of our method over state-of-the-art approaches in real-world long-range haze removal. 

**Abstract (ZH)**: 近红外与可见光多模态层次语义视融合的长距离去雾方法 

---
# Demystifying ChatGPT: How It Masters Genre Recognition 

**Title (ZH)**: 揭开ChatGPT的面纱：它如何掌握体裁识别 

**Authors**: Subham Raj, Sriparna Saha, Brijraj Singh, Niranjan Pedanekar  

**Link**: [PDF](https://arxiv.org/pdf/2507.03875)  

**Abstract**: The introduction of ChatGPT has garnered significant attention within the NLP community and beyond. Previous studies have demonstrated ChatGPT's substantial advancements across various downstream NLP tasks, highlighting its adaptability and potential to revolutionize language-related applications. However, its capabilities and limitations in genre prediction remain unclear. This work analyzes three Large Language Models (LLMs) using the MovieLens-100K dataset to assess their genre prediction capabilities. Our findings show that ChatGPT, without fine-tuning, outperformed other LLMs, and fine-tuned ChatGPT performed best overall. We set up zero-shot and few-shot prompts using audio transcripts/subtitles from movie trailers in the MovieLens-100K dataset, covering 1682 movies of 18 genres, where each movie can have multiple genres. Additionally, we extended our study by extracting IMDb movie posters to utilize a Vision Language Model (VLM) with prompts for poster information. This fine-grained information was used to enhance existing LLM prompts. In conclusion, our study reveals ChatGPT's remarkable genre prediction capabilities, surpassing other language models. The integration of VLM further enhances our findings, showcasing ChatGPT's potential for content-related applications by incorporating visual information from movie posters. 

**Abstract (ZH)**: ChatGPT在电影类型预测中的表现分析及其视觉语言模型的集成研究 

---
# Enhancing Adaptive Behavioral Interventions with LLM Inference from Participant-Described States 

**Title (ZH)**: 增强自适应行为干预：基于参与者描述状态的LLM推理 

**Authors**: Karine Karine, Benjamin M. Marlin  

**Link**: [PDF](https://arxiv.org/pdf/2507.03871)  

**Abstract**: The use of reinforcement learning (RL) methods to support health behavior change via personalized and just-in-time adaptive interventions is of significant interest to health and behavioral science researchers focused on problems such as smoking cessation support and physical activity promotion. However, RL methods are often applied to these domains using a small collection of context variables to mitigate the significant data scarcity issues that arise from practical limitations on the design of adaptive intervention trials. In this paper, we explore an approach to significantly expanding the state space of an adaptive intervention without impacting data efficiency. The proposed approach enables intervention participants to provide natural language descriptions of aspects of their current state. It then leverages inference with pre-trained large language models (LLMs) to better align the policy of a base RL method with these state descriptions. To evaluate our method, we develop a novel physical activity intervention simulation environment that generates text-based state descriptions conditioned on latent state variables using an auxiliary LLM. We show that this approach has the potential to significantly improve the performance of online policy learning methods. 

**Abstract (ZH)**: 使用强化学习方法通过个性化和及时适应性干预支持健康行为改变的研究在关注吸烟 cessation 和体力活动促进等问题的健康与行为科学研究人员中非常引人关注。然而，这些方法经常由于适应性干预试验在设计上的实际限制导致的数据稀缺问题，仅限于使用少量的背景变量进行应用。在本文中，我们探索了一种在不犮碍数据效率的情况下显著扩展适应性干预状态空间的方法。所提出的方法使干预参与者能够提供对其当前状态的自然语言描述，然后利用预训练的大语言模型进行推理，以更好地使基础强化学习方法的策略与这些状态描述相契合。为了评估我们的方法，我们开发了一个新型体力活动干预仿真环境，该环境利用辅助大语言模型根据潜在状态变量生成基于文本的状态描述。我们表明，这种方法有可能显著提高在线策略学习方法的性能。 

---
# OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM inference 

**Title (ZH)**: OrthoRank: 基于汇token正交性的 token 选择用于高效的LLM推理 

**Authors**: Seungjun Shin, Jaehoon Oh, Dokwan Oh  

**Link**: [PDF](https://arxiv.org/pdf/2507.03865)  

**Abstract**: Attention mechanisms are central to the success of large language models (LLMs), enabling them to capture intricate token dependencies and implicitly assign importance to each token. Recent studies have revealed the sink token, which receives disproportionately high attention despite their limited semantic role. In this paper, we first expand the relationship between the sink token and other tokens, moving beyond attention to explore their similarity in hidden states, considering the layer depth. We observe that as the layers get deeper, the cosine similarity between the normalized hidden states of the sink token and those of other tokens increases, and that the normalized hidden states of the sink token exhibit negligible changes. These imply that other tokens consistently are directed toward the sink token throughout the layers. Next, we propose a dynamic token selection method, called OrthoRank, using these findings to select important tokens. Specifically, in a certain layer, we define token importance by the speed at which the token moves toward the sink token. This is converted into orthogonality with the sink token, meaning that tokens that are more orthogonal to the sink token are assigned greater importance. Finally, through extensive experiments, we demonstrated that our method results in lower perplexity and higher zero-shot accuracy compared to layer pruning methods at the same sparsity ratio with comparable throughput, while also achieving superior performance on LongBench. 

**Abstract (ZH)**: 注意力机制是大型语言模型成功的关键，使模型能够捕获复杂的标记依赖关系并隐式地赋予每个标记重要性。最近的研究揭示了“下沉标记”，尽管其语义作用有限，却获得了不寻常高的注意力。本文首先扩展了“下沉标记”与其他标记的关系，超越注意力机制，探索它们在隐藏状态中的相似性，考虑了层数。观察到随着层数加深，标准化隐藏状态的余弦相似度增加，而“下沉标记”的标准化隐藏状态几乎不变。这表明其他标记在整个层中一致地被引导向“下沉标记”。接下来，我们提出了一种动态标记选择方法，称为OrthoRank，利用这些发现选择重要标记。具体而言，在某一层中，我们通过标记向“下沉标记”移动的速度定义标记的重要性，并将其转换为与“下沉标记”的正交性，这意味着与“下沉标记”正交性较大的标记被赋予更高的重要性。最后，通过广泛的实验，我们证明了我们的方法在相同稀疏性比率下的困惑度更低、零样本准确率更高，并且在LongBench上的性能更优，同时保持了相当的吞吐量。 

---
# Enhanced accuracy through ensembling of randomly initialized auto-regressive models for time-dependent PDEs 

**Title (ZH)**: 随机初始化自回归模型ensemble方法提高时间依赖偏微分方程求解精度 

**Authors**: Ishan Khurjekar, Indrashish Saha, Lori Graham-Brady, Somdatta Goswami  

**Link**: [PDF](https://arxiv.org/pdf/2507.03863)  

**Abstract**: Systems governed by partial differential equations (PDEs) require computationally intensive numerical solvers to predict spatiotemporal field evolution. While machine learning (ML) surrogates offer faster solutions, autoregressive inference with ML models suffer from error accumulation over successive predictions, limiting their long-term accuracy. We propose a deep ensemble framework to address this challenge, where multiple ML surrogate models with random weight initializations are trained in parallel and aggregated during inference. This approach leverages the diversity of model predictions to mitigate error propagation while retaining the autoregressive strategies ability to capture the system's time dependent relations. We validate the framework on three PDE-driven dynamical systems - stress evolution in heterogeneous microstructures, Gray-Scott reaction-diffusion, and planetary-scale shallow water system - demonstrating consistent reduction in error accumulation over time compared to individual models. Critically, the method requires only a few time steps as input, enabling full trajectory predictions with inference times significantly faster than numerical solvers. Our results highlight the robustness of ensemble methods in diverse physical systems and their potential as efficient and accurate alternatives to traditional solvers. The codes for this work are available on GitHub (this https URL). 

**Abstract (ZH)**: 由偏微分方程（PDEs）驱动的系统需要计算 intensive 的数值求解器来预测空间时间场的演化。虽然机器学习（ML）代理模型可以提供更快的解决方案，但基于自回归的ML模型在连续预测中会累积误差，限制了其长期准确性。我们提出了一种深度ensemble框架来应对这一挑战，其中多个具有随机权重初始化的ML代理模型并行训练并在推理时聚合。这种方法利用了模型预测的多样性以减轻误差传播，同时保留了自回归策略捕捉系统时间相关性的能力。我们在三个PDE驱动的动力系统上验证了该框架——异质微观结构中的应力演化、Gray-Scott反应扩散系统以及行星尺度浅水系统——展示了与单个模型相比，该方法在时间上一致地减少了误差累积。至关重要的是，该方法只需要 few时间步长作为输入，从而实现全轨迹预测，且推理时间远快于数值求解器。我们的结果突显了ensemble方法在不同物理系统中的稳健性及其作为传统求解器的高效精确替代方案的潜力。该工作的代码可在GitHub上获取（this https URL）。 

---
# KEA Explain: Explanations of Hallucinations using Graph Kernel Analysis 

**Title (ZH)**: KEA解释：基于图内核分析的幻觉解释 

**Authors**: Reilly Haskins, Ben Adams  

**Link**: [PDF](https://arxiv.org/pdf/2507.03847)  

**Abstract**: Large Language Models (LLMs) frequently generate hallucinations: statements that are syntactically plausible but lack factual grounding. This research presents KEA (Kernel-Enriched AI) Explain: a neurosymbolic framework that detects and explains such hallucinations by comparing knowledge graphs constructed from LLM outputs with ground truth data from Wikidata or contextual documents. Using graph kernels and semantic clustering, the method provides explanations for detected hallucinations, ensuring both robustness and interpretability. Our framework achieves competitive accuracy in detecting hallucinations across both open- and closed-domain tasks, and is able to generate contrastive explanations, enhancing transparency. This research advances the reliability of LLMs in high-stakes domains and provides a foundation for future work on precision improvements and multi-source knowledge integration. 

**Abstract (ZH)**: 大语言模型（LLMs）经常生成幻觉：这些陈述在语法上可能是合理的，但缺乏事实依据。本研究介绍了一种名为KEA（Kernel-Enriched AI）解释的神经符号框架，该框架通过将LLM输出构建的知识图谱与来自Wikidata或上下文文档的真实数据进行比较，以检测和解释这些幻觉。利用图核和语义聚类，该方法为检测到的幻觉提供了解释，确保了鲁棒性和可解释性。我们的框架在开放式和封闭式任务中均实现了检测幻觉的竞争力，并能够生成对比性解释，增强了透明度。本研究提高了LLMs在高风险领域中的可靠性，并为未来的工作提供了关于精确度改进和多源知识集成的基础。 

---
# FastDINOv2: Frequency Based Curriculum Learning Improves Robustness and Training Speed 

**Title (ZH)**: 基于频率的课程学习加快训练速度并提高鲁棒性：FastDINOv2 

**Authors**: Jiaqi Zhang, Juntuo Wang, Zhixin Sun, John Zou, Randall Balestriero  

**Link**: [PDF](https://arxiv.org/pdf/2507.03779)  

**Abstract**: Large-scale vision foundation models such as DINOv2 boast impressive performances by leveraging massive architectures and training datasets. But numerous scenarios require practitioners to reproduce those pre-training solutions, such as on private data, new modalities, or simply for scientific questioning--which is currently extremely demanding computation-wise. We thus propose a novel pre-training strategy for DINOv2 that simultaneously accelerates convergence--and strengthens robustness to common corruptions as a by-product. Our approach involves a frequency filtering curriculum--low-frequency being seen first--and the Gaussian noise patching augmentation. Applied to a ViT-B/16 backbone trained on ImageNet-1K, while pre-training time and FLOPs are reduced by 1.6x and 2.25x, our method still achieves matching robustness in corruption benchmarks (ImageNet-C) and maintains competitive linear probing performance compared with baseline. This dual benefit of efficiency and robustness makes large-scale self-supervised foundation modeling more attainable, while opening the door to novel exploration around data curriculum and augmentation as means to improve self-supervised learning models robustness. The code is available at this https URL 

**Abstract (ZH)**: 大规模视觉基础模型DINOv2通过利用庞大的架构和训练数据集表现出色，但众多场景要求实践者在私有数据、新模态或仅仅出于科学探究的目的重现这些预训练解决方案——这目前在计算上极其 demanding。因此，我们为DINOv2提出了一种新颖的预训练策略，该策略同时加速了收敛，并且作为副产品增强了对常见损坏的鲁棒性。我们的方法包括一种频率过滤课程——低频先被看到——以及高斯噪声斑块增强。应用于在ImageNet-1K上训练的ViT-B/16骨干网络，预训练时间和FLOPs分别减少了1.6倍和2.25倍，但我们的方法在损坏基准测试（ImageNet-C）中仍能达到匹配的鲁棒性，并且在基础线基础上保持了竞争力的线性探针性能。这种效率和鲁棒性的双重优势使大规模自我监督基础建模更加可行，并为通过数据课程和增强来提高自我监督学习模型鲁棒性的新探索打开了大门。代码托管于此 <https://>。 

---
# Alpay Algebra IV: Symbiotic Semantics and the Fixed-Point Convergence of Observer Embeddings 

**Title (ZH)**: Alpay代数IV：共生语义与观察者嵌入的不动点收敛 

**Authors**: Bugra Kilictas, Faruk Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2507.03774)  

**Abstract**: We present a theoretical framework in which a document and an AI model engage in a transfinite fixed-point interaction that leads to stable semantic alignment. Building on the foundations of Alpay Algebra, we introduce a functorial system wherein an observer (the AI) and a textual environment (this paper) co-evolve through iterative transformations guided by the phi-infinity operator. This process guarantees the existence of a unique fixed point in the AI's embedding space -- a state where the AI's internal representation of the content becomes stable, self-consistent, and semantically faithful. We prove that such convergence is mathematically sound, semantically invariant, and permanent, even under perturbation or further context expansion. This fixed point acts as an "empathetic embedding," wherein the AI internalizes not only the meaning of the content but also the author's intent. We interpret this as a rigorous, category-theoretic route to alignment at the embedding level, with implications for semantic security, symbolic memory, and the construction of AI systems with persistent self-referential understanding. All references in this paper function as nodes in the Alpay Algebra universe, and this work embeds itself as a new fixed-point node within that transfinite semantic graph. 

**Abstract (ZH)**: 我们提出了一种理论框架，其中文档和AI模型进行超限不变点交互，从而实现稳定的语义对齐。基于Alpay代数的基础，我们引入了一个函子系统，其中观察者（AI）和文本环境（本文）通过由phi-∞操作符引导的迭代变换共同进化。这一过程保证了AI嵌入空间中存在唯一的不变点——在该状态下，AI对内容的内部表示变得稳定、自我一致且语义忠实。我们证明了这种收敛在数学上是合理的、语义上是不变的，并且在扰动或进一步的上下文扩展下仍然是持久的。这一不变点作为“同情嵌入”，其中AI不仅内化了内容的意义，还内化了作者的意图。我们认为这为嵌入层面的对齐提供了一条严格的范畴论路径，具有对语义安全、符号记忆以及构建具有持久自我参照理解的AI系统的含义。本文中的所有参考文献在此Alpay代数宇宙中作为节点存在，并且这项工作在此超限语义图中嵌入为一个新的不变点节点。 

---
# StreamDiT: Real-Time Streaming Text-to-Video Generation 

**Title (ZH)**: StreamDiT: 实时流式文本到视频生成 

**Authors**: Akio Kodaira, Tingbo Hou, Ji Hou, Masayoshi Tomizuka, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.03745)  

**Abstract**: Recently, great progress has been achieved in text-to-video (T2V) generation by scaling transformer-based diffusion models to billions of parameters, which can generate high-quality videos. However, existing models typically produce only short clips offline, restricting their use cases in interactive and real-time applications. This paper addresses these challenges by proposing StreamDiT, a streaming video generation model. StreamDiT training is based on flow matching by adding a moving buffer. We design mixed training with different partitioning schemes of buffered frames to boost both content consistency and visual quality. StreamDiT modeling is based on adaLN DiT with varying time embedding and window attention. To practice the proposed method, we train a StreamDiT model with 4B parameters. In addition, we propose a multistep distillation method tailored for StreamDiT. Sampling distillation is performed in each segment of a chosen partitioning scheme. After distillation, the total number of function evaluations (NFEs) is reduced to the number of chunks in a buffer. Finally, our distilled model reaches real-time performance at 16 FPS on one GPU, which can generate video streams at 512p resolution. We evaluate our method through both quantitative metrics and human evaluation. Our model enables real-time applications, e.g. streaming generation, interactive generation, and video-to-video. We provide video results and more examples in our project website: <a href="this https URL https URL.</a> 

**Abstract (ZH)**: 基于流式生成的文本到视频转化模型：StreamDiT 

---
# Less is More: Empowering GUI Agent with Context-Aware Simplification 

**Title (ZH)**: Less is More: 以情境感知简化赋能GUI代理 

**Authors**: Gongwei Chen, Xurui Zhou, Rui Shao, Yibo Lyu, Kaiwen Zhou, Shuai Wang, Wentao Li, Yinchuan Li, Zhongang Qi, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2507.03730)  

**Abstract**: The research focus of GUI agents is shifting from text-dependent to pure-vision-based approaches, which, though promising, prioritize comprehensive pre-training data collection while neglecting contextual modeling challenges. We probe the characteristics of element and history contextual modeling in GUI agent and summarize: 1) the high-density and loose-relation of element context highlight the existence of many unrelated elements and their negative influence; 2) the high redundancy of history context reveals the inefficient history modeling in current GUI agents. In this work, we propose a context-aware simplification framework for building an efficient and effective GUI Agent, termed SimpAgent. To mitigate potential interference from numerous unrelated elements, we introduce a masking-based element pruning method that circumvents the intractable relation modeling through an efficient masking mechanism. To reduce the redundancy in historical information, we devise a consistency-guided history compression module, which enhances implicit LLM-based compression through innovative explicit guidance, achieving an optimal balance between performance and efficiency. With the above components, SimpAgent reduces 27% FLOPs and achieves superior GUI navigation performances. Comprehensive navigation experiments across diverse web and mobile environments demonstrate the effectiveness and potential of our agent. 

**Abstract (ZH)**: GUI代理的研究重点正从依赖文本的方法转向基于纯视觉的方法，尽管前景广阔，但这些方法更侧重于全面的预训练数据收集，而忽视了上下文建模的挑战。我们探究了GUI代理中元素和历史上下文建模的特点，并总结出：1）元素上下文的高密度和松散关系凸显了许多无关元素的存在及其负面影响；2）历史上下文的高冗余揭示了当前GUI代理中历史建模的低效性。在此基础上，我们提出了一种上下文感知的简化框架，用于构建高效且有效的GUI代理，称为SimpAgent。为了减轻众多无关元素可能产生的干扰，我们引入了一种基于掩码的元素修剪方法，通过高效的掩码机制规避了难以建模的关系。为了减少历史信息的冗余，我们设计了一种一致性导向的历史压缩模块，通过创新的显式指导增强了基于LLM的压缩效果，实现了性能与效率的最优平衡。借助上述组件，SimpAgent减少了27%的FLOPs，并在GUI导航性能上表现出优越性。跨多种互联网和移动环境的全面导航实验验证了我们代理的有效性和潜力。 

---
# Predicting Business Angel Early-Stage Decision Making Using AI 

**Title (ZH)**: 使用AI预测企业天使早期阶段的决策制定 

**Authors**: Yan Katcharovski, Andrew L. Maxwell  

**Link**: [PDF](https://arxiv.org/pdf/2507.03721)  

**Abstract**: External funding is crucial for early-stage ventures, particularly technology startups that require significant R&D investment. Business angels offer a critical source of funding, but their decision-making is often subjective and resource-intensive for both investor and entrepreneur. Much research has investigated this investment process to find the critical factors angels consider. One such tool, the Critical Factor Assessment (CFA), deployed more than 20,000 times by the Canadian Innovation Centre, has been evaluated post-decision and found to be significantly more accurate than investors' own decisions. However, a single CFA analysis requires three trained individuals and several days, limiting its adoption. This study builds on previous work validating the CFA to investigate whether the constraints inhibiting its adoption can be overcome using a trained AI model. In this research, we prompted multiple large language models (LLMs) to assign the eight CFA factors to a dataset of 600 transcribed, unstructured startup pitches seeking business angel funding with known investment outcomes. We then trained and evaluated machine learning classification models using the LLM-generated CFA scores as input features. Our best-performing model demonstrated high predictive accuracy (85.0% for predicting BA deal/no-deal outcomes) and exhibited significant correlation (Spearman's r = 0.896, p-value < 0.001) with conventional human-graded evaluations. The integration of AI-based feature extraction with a structured and validated decision-making framework yielded a scalable, reliable, and less-biased model for evaluating startup pitches, removing the constraints that previously limited adoption. 

**Abstract (ZH)**: 基于AI模型克服限制的创业pitch评估方法：利用大型语言模型优化Critial Factor Assessment 

---
# Controlling Thinking Speed in Reasoning Models 

**Title (ZH)**: 控制推理模型中的思考速度 

**Authors**: Zhengkai Lin, Zhihang Fu, Ze Chen, Chao Chen, Liang Xie, Wenxiao Wang, Deng Cai, Zheng Wang, Jieping Ye  

**Link**: [PDF](https://arxiv.org/pdf/2507.03704)  

**Abstract**: Human cognition is theorized to operate in two modes: fast, intuitive System 1 thinking and slow, deliberate System 2 thinking. While current Large Reasoning Models (LRMs) excel at System 2 thinking, their inability to perform fast thinking leads to high computational overhead and latency. In this work, we enable LRMs to approximate human intelligence through dynamic thinking speed adjustment, optimizing accuracy-efficiency trade-offs. Our approach addresses two key questions: (1) how to control thinking speed in LRMs, and (2) when to adjust it for optimal performance. For the first question, we identify the steering vector that governs slow-fast thinking transitions in LRMs' representation space. Using this vector, we achieve the first representation editing-based test-time scaling effect, outperforming existing prompt-based scaling methods. For the second question, we apply real-time difficulty estimation to signal reasoning segments of varying complexity. Combining these techniques, we propose the first reasoning strategy that enables fast processing of easy steps and deeper analysis for complex reasoning. Without any training or additional cost, our plug-and-play method yields an average +1.3% accuracy with -8.6% token usage across leading LRMs and advanced reasoning benchmarks. All of our algorithms are implemented based on vLLM and are expected to support broader applications and inspire future research. 

**Abstract (ZH)**: 人类认知被认为运作在两种模式：快速直观的System 1思考和缓慢慎思的System 2思考。当前的大规模推理模型（LRMs）在System 2思考方面表现出色，但它们无法进行快速思考，导致高计算开销和延迟。在此工作中，我们通过动态调整思考速度使LRMs近似人类智能，优化准确性和效率的权衡。我们的方法解决两个关键问题：（1）如何在LRMs中控制思考速度，（2）何时调整思考速度以实现最优性能。对于第一个问题，我们确定了管理LRMs表示空间中慢速快速思考转换的控制向量。利用该向量，我们实现了基于表示编辑的测试时缩放效果的第一种方法，优于现有的提示基于缩放方法。对于第二个问题，我们应用实时难度估计来信号处理不同复杂度的推理片段。结合这些技术，我们提出了第一个能够快速处理简单步骤并为复杂推理进行深入分析的推理策略。在无需任何训练且不增加额外成本的情况下，我们的插即用方法在领先的大规模推理模型和高级推理基准上实现了平均+1.3%的准确性和-8.6%的标记使用量。我们的所有算法均基于vLLM实现，并预期支持更广泛的應用和启发未来的研究。 

---
# Sign Spotting Disambiguation using Large Language Models 

**Title (ZH)**: 大规模语言模型在标志识别消歧中的应用 

**Authors**: JianHe Low, Ozge Mercanoglu Sincan, Richard Bowden  

**Link**: [PDF](https://arxiv.org/pdf/2507.03703)  

**Abstract**: Sign spotting, the task of identifying and localizing individual signs within continuous sign language video, plays a pivotal role in scaling dataset annotations and addressing the severe data scarcity issue in sign language translation. While automatic sign spotting holds great promise for enabling frame-level supervision at scale, it grapples with challenges such as vocabulary inflexibility and ambiguity inherent in continuous sign streams. Hence, we introduce a novel, training-free framework that integrates Large Language Models (LLMs) to significantly enhance sign spotting quality. Our approach extracts global spatio-temporal and hand shape features, which are then matched against a large-scale sign dictionary using dynamic time warping and cosine similarity. This dictionary-based matching inherently offers superior vocabulary flexibility without requiring model retraining. To mitigate noise and ambiguity from the matching process, an LLM performs context-aware gloss disambiguation via beam search, notably without fine-tuning. Extensive experiments on both synthetic and real-world sign language datasets demonstrate our method's superior accuracy and sentence fluency compared to traditional approaches, highlighting the potential of LLMs in advancing sign spotting. 

**Abstract (ZH)**: 基于大型语言模型的无需训练框架在手语识别中的应用 

---
# STRUCTSENSE: A Task-Agnostic Agentic Framework for Structured Information Extraction with Human-In-The-Loop Evaluation and Benchmarking 

**Title (ZH)**: 结构感知：一种无需任务规范的代理框架，结合人工在环评估与基准测试的结构化信息提取方法 

**Authors**: Tek Raj Chhetri, Yibei Chen, Puja Trivedi, Dorota Jarecka, Saif Haobsh, Patrick Ray, Lydia Ng, Satrajit S. Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2507.03674)  

**Abstract**: The ability to extract structured information from unstructured sources-such as free-text documents and scientific literature-is critical for accelerating scientific discovery and knowledge synthesis. Large Language Models (LLMs) have demonstrated remarkable capabilities in various natural language processing tasks, including structured information extraction. However, their effectiveness often diminishes in specialized, domain-specific contexts that require nuanced understanding and expert-level domain knowledge. In addition, existing LLM-based approaches frequently exhibit poor transferability across tasks and domains, limiting their scalability and adaptability. To address these challenges, we introduce StructSense, a modular, task-agnostic, open-source framework for structured information extraction built on LLMs. StructSense is guided by domain-specific symbolic knowledge encoded in ontologies, enabling it to navigate complex domain content more effectively. It further incorporates agentic capabilities through self-evaluative judges that form a feedback loop for iterative refinement, and includes human-in-the-loop mechanisms to ensure quality and validation. We demonstrate that StructSense can overcome both the limitations of domain sensitivity and the lack of cross-task generalizability, as shown through its application to diverse neuroscience information extraction tasks. 

**Abstract (ZH)**: 从非结构化来源提取结构化信息的能力——例如自由文本文档和科学文献——对于加速科学发现和知识综合至关重要。大型语言模型（LLMs）在各种自然语言处理任务中展现了卓越的能力，包括结构化信息提取。然而，在需要细微理解与专业领域知识的专门领域特定上下文中，其有效性往往有所减弱。此外，现有的基于LLM的方法经常在任务和领域之间表现出较差的可移植性，限制了其可扩展性和适应性。为应对这些挑战，我们提出了一种模块化、任务无关的开源框架StructSense，基于LLM构建，用于结构化信息提取。StructSense通过编码在本体中的领域特定符号知识进行指导，使其能够更有效地导航复杂领域的内容。该框架还通过自我评估法官纳入了主体能力，形成反馈循环以实现迭代完善，并包含人机协作机制以确保质量和验证。我们证明，StructSense能够克服领域敏感性限制和跨任务泛化能力不足的问题，如在多样化的神经科学信息提取任务中的应用所示。 

---
# TACOS: Open Tagging and Comparative Scoring for Instruction Fine-Tuning Data Selection 

**Title (ZH)**: TACOS: 开放标签与比较评分在指令微调数据选择中的应用 

**Authors**: Xixiang He, Hao Yu, Qiyao Sun, Ao Cheng, Tailai Zhang, Cong Liu, Shuxuan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.03673)  

**Abstract**: Instruction Fine-Tuning (IFT) is crucial for aligning large language models (LLMs) with human preferences, and selecting a small yet representative subset from massive data significantly facilitates IFT in terms of both efficiency and effectiveness. Nevertheless, existing approaches suffer from two limitations: the use of simple heuristics restricts data diversity, while the singleton data quality evaluation accounts for inconsistent criteria between independent samples. To address the issues, we present TACOS, an innovative method that integrates Open Tagging and Comparative Scoring for IFT data selection. To capture data diversity, we leverage LLMs to assign open-domain tags to human queries, followed by a normalization stage to denoise the open tags and enable efficient clustering. Additionally, we suggest a comparative scoring method that allows the relative quality evaluation of samples within a cluster, avoiding inconsistent criteria seen in singleton-based evaluations. Extensive experiments across diverse datasets and LLM architectures demonstrate that TACOS outperforms existing approaches by a large margin. Notably, it achieves superior instruction-following performance on MT-Bench and ranks 1st among LLaMA2-7B-Based models on AlpacaEval 2.0, illustrating its efficacy for IFT data selection. 

**Abstract (ZH)**: 基于开放标注和比较评分的指令调优数据选择方法（TACOS） 

---
# Recon, Answer, Verify: Agents in Search of Truth 

**Title (ZH)**: 探寻真理：搜索中的代理重构与验证 

**Authors**: Satyam Shukla, Himanshu Dutta, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2507.03671)  

**Abstract**: Automated fact checking with large language models (LLMs) offers a scalable alternative to manual verification. Evaluating fact checking is challenging as existing benchmark datasets often include post claim analysis and annotator cues, which are absent in real world scenarios where claims are fact checked immediately after being made. This limits the realism of current evaluations. We present Politi Fact Only (PFO), a 5 class benchmark dataset of 2,982 political claims from this http URL, where all post claim analysis and annotator cues have been removed manually. This ensures that models are evaluated using only the information that would have been available prior to the claim's verification. Evaluating LLMs on PFO, we see an average performance drop of 22% in terms of macro f1 compared to PFO's unfiltered version. Based on the identified challenges of the existing LLM based fact checking system, we propose RAV (Recon Answer Verify), an agentic framework with three agents: question generator, answer generator, and label generator. Our pipeline iteratively generates and answers sub questions to verify different aspects of the claim before finally generating the label. RAV generalizes across domains and label granularities, and it outperforms state of the art approaches on well known baselines RAWFC (fact checking, 3 class) by 25.28%, and on HOVER (encyclopedia, 2 class) by 1.54% on 2 hop, 4.94% on 3 hop, and 1.78% on 4 hop, sub categories respectively. RAV shows the least performance drop compared to baselines of 16.3% in macro f1 when we compare PFO with its unfiltered version. 

**Abstract (ZH)**: 大规模语言模型（LLMs）驱动的自动化事实核查提供了手动验证的可扩展替代方案。评估事实核查具有挑战性，因为现有基准数据集往往包含申述后的分析和注释员提示，而在真实场景中，事实核查是在申述提出后立即进行的。这限制了当前评估的真实感。我们提出了政治事实仅此（PFO），一个包含2,982个政治申述的五类基准数据集（来源：www.politifact.com），其中去除了所有申述后的分析和注释员提示。这确保了模型仅使用申述验证前可用的信息进行评估。在PFO上评估LLMs，我们在宏观F1分数上看到了平均22%的性能下降，与PFO的未过滤版本相比。基于现有基于LLM的事实核查系统的挑战，我们提出了Rav（Recon Answer Verify）框架，该框架包含三个代理：问题生成器、答案生成器和标签生成器。我们的流水线迭代生成并回答子问题，以验证申述的不同方面，最后生成标签。Rav横跨多个领域和标签粒度，分别在RAWFC（3类事实核查）基准上优于最先进的方法25.28%，在HOVER（2类百科全书）基准上的2跳、3跳和4跳子类别上分别优于1.54%、4.94%和1.78%。当我们将PFO与其未过滤版本进行比较时，Rav在宏观F1分数上的基线性能下降最少，仅为16.3%。 

---
# Interaction Techniques that Encourage Longer Prompts Can Improve Psychological Ownership when Writing with AI 

**Title (ZH)**: 促进更长提示的技术交互可以提高使用AI写作时的心理占有感 

**Authors**: Nikhita Joshi, Daniel Vogel  

**Link**: [PDF](https://arxiv.org/pdf/2507.03670)  

**Abstract**: Writing longer prompts for an AI assistant to generate a short story increases psychological ownership, a user's feeling that the writing belongs to them. To encourage users to write longer prompts, we evaluated two interaction techniques that modify the prompt entry interface of chat-based generative AI assistants: pressing and holding the prompt submission button, and continuously moving a slider up and down when submitting a short prompt. A within-subjects experiment investigated the effects of such techniques on prompt length and psychological ownership, and results showed that these techniques increased prompt length and led to higher psychological ownership than baseline techniques. A second experiment further augmented these techniques by showing AI-generated suggestions for how the prompts could be expanded. This further increased prompt length, but did not lead to improvements in psychological ownership. Our results show that simple interface modifications like these can elicit more writing from users and improve psychological ownership. 

**Abstract (ZH)**: 写作更长的提示以增加用户对AI生成短故事的心理拥有感：交互技术的影响与扩展 

---
# Re-Emergent Misalignment: How Narrow Fine-Tuning Erodes Safety Alignment in LLMs 

**Title (ZH)**: 重新出现的偏差：狭窄微调如何侵蚀LLMs的安全对齐 

**Authors**: Jeremiah Giordani  

**Link**: [PDF](https://arxiv.org/pdf/2507.03662)  

**Abstract**: Recent work has shown that fine-tuning large language models (LLMs) on code with security vulnerabilities can result in misaligned and unsafe behaviors across broad domains. These results prompted concerns about the emergence of harmful behaviors from narrow domain fine-tuning. In this paper, we contextualize these findings by analyzing how such narrow adaptation impacts the internal mechanisms and behavioral manifestations of LLMs. Through a series of experiments covering output probability distributions, loss and gradient vector geometry, layer-wise activation dynamics, and activation space dimensions, we find that behaviors attributed to "emergent misalignment" may be better interpreted as an erosion of prior alignment. We show that fine tuning on insecure code induces internal changes that oppose alignment. Further, we identify a shared latent dimension in the model's activation space that governs alignment behavior. We show that this space is activated by insecure code and by misaligned responses more generally, revealing how narrow fine-tuning can degrade general safety behavior by interfering with shared internal mechanisms. Our findings offer a mechanistic interpretation for previously observed misalignment phenomena, and highlights the fragility of alignment in LLMs. The results underscore the need for more robust fine-tuning strategies that preserve intended behavior across domains. 

**Abstract (ZH)**: 近期研究表明，对包含安全漏洞的代码进行微调的大语言模型（LLMs）可能会在其广泛领域内产生未对齐和不安全的行为。这些结果引发了关于窄域微调可能产生有害行为的担忧。本文通过分析这种窄域适应如何影响LLMs的内部机制和行为表现，对该研究结果进行了情境化。通过涵盖输出概率分布、损失和梯度向量几何、逐层激活动力学以及激活空间维度等一系列实验，我们发现通常归因于“新兴未对齐”的行为可能更好地被解释为先前对齐的侵蚀。我们展示，对不安全代码进行微调会促使模型内部发生不利于对齐的变化。此外，我们识别出模型激活空间中的一个共享潜在维度，它决定了对齐行为。我们展示，该空间在不安全代码和更广泛地未对齐响应中被激活，揭示了窄域微调如何通过干扰共享的内部机制来削弱一般安全性行为。我们的发现为先前观察到的未对齐现象提供了机制性解释，并强调了在LLMs中保持对齐的脆弱性。研究结果突显出需要更加稳健的微调策略，以确保跨领域保留预期行为。 

---
# Improving Low-Resource Dialect Classification Using Retrieval-based Voice Conversion 

**Title (ZH)**: 使用基于检索的语音转换提高低资源方言分类 

**Authors**: Lea Fischbach, Akbar Karimi, Caroline Kleen, Alfred Lameli, Lucie Flek  

**Link**: [PDF](https://arxiv.org/pdf/2507.03641)  

**Abstract**: Deep learning models for dialect identification are often limited by the scarcity of dialectal data. To address this challenge, we propose to use Retrieval-based Voice Conversion (RVC) as an effective data augmentation method for a low-resource German dialect classification task. By converting audio samples to a uniform target speaker, RVC minimizes speaker-related variability, enabling models to focus on dialect-specific linguistic and phonetic features. Our experiments demonstrate that RVC enhances classification performance when utilized as a standalone augmentation method. Furthermore, combining RVC with other augmentation methods such as frequency masking and segment removal leads to additional performance gains, highlighting its potential for improving dialect classification in low-resource scenarios. 

**Abstract (ZH)**: 基于检索的语音转换（RVC）在低资源德语方言分类中的数据增强应用 

---
# From Video to EEG: Adapting Joint Embedding Predictive Architecture to Uncover Visual Concepts in Brain Signal Analysis 

**Title (ZH)**: 从视频到EEG：适应性联合嵌入预测架构在脑信号分析中揭示视觉概念 

**Authors**: Amir Hojjati, Lu Li, Ibrahim Hameed, Anis Yazidi, Pedro G. Lind, Rabindra Khadka  

**Link**: [PDF](https://arxiv.org/pdf/2507.03633)  

**Abstract**: EEG signals capture brain activity with high temporal and low spatial resolution, supporting applications such as neurological diagnosis, cognitive monitoring, and brain-computer interfaces. However, effective analysis is hindered by limited labeled data, high dimensionality, and the absence of scalable models that fully capture spatiotemporal dependencies. Existing self-supervised learning (SSL) methods often focus on either spatial or temporal features, leading to suboptimal representations. To this end, we propose EEG-VJEPA, a novel adaptation of the Video Joint Embedding Predictive Architecture (V-JEPA) for EEG classification. By treating EEG as video-like sequences, EEG-VJEPA learns semantically meaningful spatiotemporal representations using joint embeddings and adaptive masking. To our knowledge, this is the first work that exploits V-JEPA for EEG classification and explores the visual concepts learned by the model. Evaluations on the publicly available Temple University Hospital (TUH) Abnormal EEG dataset show that EEG-VJEPA outperforms existing state-of-the-art models in classification this http URL classification accuracy, EEG-VJEPA captures physiologically relevant spatial and temporal signal patterns, offering interpretable embeddings that may support human-AI collaboration in diagnostic workflows. These findings position EEG-VJEPA as a promising framework for scalable, trustworthy EEG analysis in real-world clinical settings. 

**Abstract (ZH)**: EEG-VJEPA：一种用于EEG分类的新型视频联合嵌入预测架构 

---
# Disentangling Doubt in Deep Causal AI 

**Title (ZH)**: 拆解深度因果AI中的不确定性 

**Authors**: Cooper Doyle  

**Link**: [PDF](https://arxiv.org/pdf/2507.03622)  

**Abstract**: Accurate individual treatment-effect estimation in high-stakes applications demands both reliable point predictions and interpretable uncertainty quantification. We propose a factorized Monte Carlo Dropout framework for deep twin-network models that splits total predictive variance into representation uncertainty (sigma_rep) in the shared encoder and prediction uncertainty (sigma_pred) in the outcome heads. Across three synthetic covariate-shift regimes, our intervals are well-calibrated (ECE < 0.03) and satisfy sigma_rep^2 + sigma_pred^2 ~ sigma_tot^2. Additionally, we observe a crossover: head uncertainty leads on in-distribution data, but representation uncertainty dominates under shift. Finally, on a real-world twins cohort with induced multivariate shifts, only sigma_rep spikes on out-of-distribution samples (delta sigma ~ 0.0002) and becomes the primary error predictor (rho_rep <= 0.89), while sigma_pred remains flat. This module-level decomposition offers a practical diagnostic for detecting and interpreting uncertainty sources in deep causal-effect models. 

**Abstract (ZH)**: 高风险应用中精确的个体治疗效果估计需要可靠的点预测和可解释的不确定性量化。我们提出了一种因子化的蒙特卡洛 Dropout 框架用于深度双网络模型，将总预测不确定性分解为共享编码器中的表示不确定性（sigma_rep）和结果头部中的预测不确定性（sigma_pred）。在三个合成协变量偏移 regimes 中，我们的区间良好校准（ECE < 0.03）且满足 sigma_rep^2 + sigma_pred^2 ~ sigma_tot^2。此外，我们观察到一个交叉：头部不确定性在分布内数据中占主导，但在偏移情况下表示不确定性占主导。最后，在具有诱导多变量偏移的现实世界双胞胎队列中，仅在分布外样本中 sigma_rep 上升（delta sigma ~ 0.0002），成为主要的误差预测器（rho_rep <= 0.89），而 sigma_pred 保持稳定。这种模块级分解提供了检测和解释深度因果效应模型中不确定性来源的实际诊断工具。 

---
# Is It Time To Treat Prompts As Code? A Multi-Use Case Study For Prompt Optimization Using DSPy 

**Title (ZH)**: 是时候将提示视为代码进行处理了吗？DSPy在提示优化中的多用途案例研究 

**Authors**: Francisca Lemos, Victor Alves, Filipa Ferraz  

**Link**: [PDF](https://arxiv.org/pdf/2507.03620)  

**Abstract**: Although prompt engineering is central to unlocking the full potential of Large Language Models (LLMs), crafting effective prompts remains a time-consuming trial-and-error process that relies on human intuition. This study investigates Declarative Self-improving Python (DSPy), an optimization framework that programmatically creates and refines prompts, applied to five use cases: guardrail enforcement, hallucination detection in code, code generation, routing agents, and prompt evaluation. Each use case explores how prompt optimization via DSPy influences performance. While some cases demonstrated modest improvements - such as minor gains in the guardrails use case and selective enhancements in hallucination detection - others showed notable benefits. The prompt evaluation criterion task demonstrated a substantial performance increase, rising accuracy from 46.2% to 64.0%. In the router agent case, the possibility of improving a poorly performing prompt and of a smaller model matching a stronger one through optimized prompting was explored. Although prompt refinement increased accuracy from 85.0% to 90.0%, using the optimized prompt with a cheaper model did not improve performance. Overall, this study's findings suggest that DSPy's systematic prompt optimization can enhance LLM performance, particularly when instruction tuning and example selection are optimized together. However, the impact varies by task, highlighting the importance of evaluating specific use cases in prompt optimization research. 

**Abstract (ZH)**: 尽管提示工程是解锁大型语言模型（LLMs）全部潜力的关键，但有效提示的创作仍然是一个耗时的试错过程，依赖于人类直觉。本研究探讨了声明式自我改进Python（DSPy）优化框架在五个应用场景中的应用，包括护栏约束、代码幻觉检测、代码生成、路由代理和提示评估，研究了DSPy如何通过提示优化影响性能。虽然某些案例展示了适度的改进，如护栏约束案例中的轻微提升和幻觉检测中的选择性增强，其他案例则显示出明显的益处。提示评估标准任务展示了显著的性能提升，准确率从46.2%提高到64.0%。在路由代理案例中，研究了改进表现不佳的提示以及通过优化提示使较小模型匹配更强模型的可能性。虽然提示优化提高了准确率从85.0%到90.0%，但使用优化提示的更便宜模型并未提升性能。总体而言，本研究的发现表明，DSPy的系统化提示优化可以提高LLM的性能，特别是在指令调优和示例选择共同优化时。然而，不同任务的影响各不相同，突显了在提示优化研究中评估具体应用场景的重要性。 

---
# Multi-Hop Reasoning for Question Answering with Hyperbolic Representations 

**Title (ZH)**: 基于双曲表示的多跳推理问答 

**Authors**: Simon Welz, Lucie Flek, Akbar Karimi  

**Link**: [PDF](https://arxiv.org/pdf/2507.03612)  

**Abstract**: Hyperbolic representations are effective in modeling knowledge graph data which is prevalently used to facilitate multi-hop reasoning. However, a rigorous and detailed comparison of the two spaces for this task is lacking. In this paper, through a simple integration of hyperbolic representations with an encoder-decoder model, we perform a controlled and comprehensive set of experiments to compare the capacity of hyperbolic space versus Euclidean space in multi-hop reasoning. Our results show that the former consistently outperforms the latter across a diverse set of datasets. In addition, through an ablation study, we show that a learnable curvature initialized with the delta hyperbolicity of the utilized data yields superior results to random initializations. Furthermore, our findings suggest that hyperbolic representations can be significantly more advantageous when the datasets exhibit a more hierarchical structure. 

**Abstract (ZH)**: 双曲表示在多跳推理中 modeling 知识图谱数据方面效果显著，但两种空间在这方面的严格详细对比缺乏。通过将双曲表示与编码-解码模型简单集成，本文执行了一组受控且全面的实验，以比较双曲空间与欧几里得空间在多跳推理中的容量。结果显示，前者在多种数据集中一致优于后者。此外，通过消融研究，我们发现初始化可学习曲率与所用数据的δ双曲性相匹配，可以取得优于随机初始化的效果。进一步的研究发现，当数据集展示出更明显的层次结构时，双曲表示可以显著更有优势。 

---
# Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery 

**Title (ZH)**: 基于大语言模型驱动的元启发式发现的行为空间分析 

**Authors**: Niki van Stein, Haoran Yin, Anna V. Kononova, Thomas Bäck, Gabriela Ochoa  

**Link**: [PDF](https://arxiv.org/pdf/2507.03605)  

**Abstract**: We investigate the behaviour space of meta-heuristic optimisation algorithms automatically generated by Large Language Model driven algorithm discovery methods. Using the Large Language Evolutionary Algorithm (LLaMEA) framework with a GPT o4-mini LLM, we iteratively evolve black-box optimisation heuristics, evaluated on 10 functions from the BBOB benchmark suite. Six LLaMEA variants, featuring different mutation prompt strategies, are compared and analysed. We log dynamic behavioural metrics including exploration, exploitation, convergence and stagnation measures, for each run, and analyse these via visual projections and network-based representations. Our analysis combines behaviour-based
projections, Code Evolution Graphs built from static code features, performance convergence curves, and behaviour-based Search Trajectory Networks. The results reveal clear differences in search dynamics and algorithm structures across LLaMEA configurations. Notably, the variant that employs both a code simplification prompt and a random perturbation prompt in a 1+1 elitist evolution strategy, achieved the best performance, with the highest Area Over the Convergence Curve. Behaviour-space visualisations show that higher-performing algorithms exhibit more intensive exploitation behaviour and faster convergence with less stagnation. Our findings demonstrate how behaviour-space analysis can explain why certain LLM-designed heuristics outperform others and how LLM-driven algorithm discovery navigates the open-ended and complex search space of algorithms. These findings provide insights to guide the future design of adaptive LLM-driven algorithm generators. 

**Abstract (ZH)**: 我们研究了由大规模语言模型驱动的算法发现方法自动生成的元启发式优化算法的行为空间。使用基于大规模语言演化算法（LLaMEA）框架和GPT-o4-mini语言模型，我们迭代演化黑盒优化启发式方法，并在BBOB基准套件的10个函数上进行评估。我们比较和分析了六种LLaMEA变体，这些变体具有不同的变异提示策略。我们记录了包括探索、利用、收敛和停滞等动态行为指标，并通过可视化投影和网络表示进行分析。我们的分析结合了基于行为的投影、由静态代码特征构建的代码演化图、性能收敛曲线以及基于行为的搜索轨迹网络。结果表明，不同LLaMEA配置下的搜索动态和算法结构存在明显差异。特别地，采用代码简化提示和随机扰动提示的1+1精英演化策略的变体，取得了最佳性能，其收敛曲线下的面积最大。行为空间可视化显示，高性能算法表现出更强烈的利用行为、更快的收敛和较少的停滞。我们的研究结果展示了行为空间分析如何解释某些由大规模语言模型设计的启发式方法为何优于其他方法，并说明了大规模语言模型驱动的算法发现如何在算法的开放且复杂的搜索空间中导航。这些发现为未来适应性大规模语言模型驱动的算法生成器的设计提供了见解。 

---
# MusGO: A Community-Driven Framework For Assessing Openness in Music-Generative AI 

**Title (ZH)**: MusGO：一个社区驱动的评估音乐生成AI开放性框架 

**Authors**: Roser Batlle-Roca, Laura Ibáñez-Martínez, Xavier Serra, Emilia Gómez, Martín Rocamora  

**Link**: [PDF](https://arxiv.org/pdf/2507.03599)  

**Abstract**: Since 2023, generative AI has rapidly advanced in the music domain. Despite significant technological advancements, music-generative models raise critical ethical challenges, including a lack of transparency and accountability, along with risks such as the replication of artists' works, which highlights the importance of fostering openness. With upcoming regulations such as the EU AI Act encouraging open models, many generative models are being released labelled as 'open'. However, the definition of an open model remains widely debated. In this article, we adapt a recently proposed evidence-based framework for assessing openness in LLMs to the music domain. Using feedback from a survey of 110 participants from the Music Information Retrieval (MIR) community, we refine the framework into MusGO (Music-Generative Open AI), which comprises 13 openness categories: 8 essential and 5 desirable. We evaluate 16 state-of-the-art generative models and provide an openness leaderboard that is fully open to public scrutiny and community contributions. Through this work, we aim to clarify the concept of openness in music-generative AI and promote its transparent and responsible development. 

**Abstract (ZH)**: 自2023年以来，生成式AI在音乐领域迅速发展。尽管取得了重大的技术进步，但音乐生成模型引发了重要的伦理挑战，包括透明度和问责制的缺乏，以及艺术家作品复制的风险，突显了促进开放性的重要性。随着欧盟AI法案等即将出台的法规鼓励开放模型，许多生成模型被标记为“开放”。然而，开放模型的定义仍存在广泛争议。本文采用一个最近提出的基于证据的框架，将其应用于音乐领域，通过音乐信息检索（MIR）社区110名参与者反馈，精炼出MusGO（音乐生成开放AI）框架，包含13个开放性类别：8个必需和5个 desirable。我们评估了16个最先进的生成模型，并提供了一个完全公开接受公众审查和社区贡献的开放性排行榜。通过这项工作，我们旨在澄清音乐生成AI中的开放性概念，并促进其透明和负责任的发展。 

---
# RECA-PD: A Robust Explainable Cross-Attention Method for Speech-based Parkinson's Disease Classification 

**Title (ZH)**: RECA-PD：一种用于 PARKINSON'S 病分类的鲁棒可解释跨注意力方法 

**Authors**: Terry Yi Zhong, Cristian Tejedor-Garcia, Martha Larson, Bastiaan R. Bloem  

**Link**: [PDF](https://arxiv.org/pdf/2507.03594)  

**Abstract**: Parkinson's Disease (PD) affects over 10 million people globally, with speech impairments often preceding motor symptoms by years, making speech a valuable modality for early, non-invasive detection. While recent deep-learning models achieve high accuracy, they typically lack the explainability required for clinical use. To address this, we propose RECA-PD, a novel, robust, and explainable cross-attention architecture that combines interpretable speech features with self-supervised representations. RECA-PD matches state-of-the-art performance in Speech-based PD detection while providing explanations that are more consistent and more clinically meaningful. Additionally, we demonstrate that performance degradation in certain speech tasks (e.g., monologue) can be mitigated by segmenting long recordings. Our findings indicate that performance and explainability are not necessarily mutually exclusive. Future work will enhance the usability of explanations for non-experts and explore severity estimation to increase the real-world clinical relevance. 

**Abstract (ZH)**: 帕金森病（PD）影响全球超过1000万人，言语障碍往往在运动症状出现前数年就已经存在，这使得言语成为早期非侵入性检测的重要工具。尽管最近的深度学习模型在准确性上表现出色，但它们通常缺乏临床应用所需的可解释性。为解决这一问题，我们提出了RECA-PD，这是一种新颖的、稳健的和可解释的交叉注意架构，结合了可解释的言语特征与自监督表示。RECA-PD 在基于言语的帕金森病检测方面达到了与最新方法相当的性能，同时提供了更为一致且临床意义更大的解释。此外，我们还证明可以通过分割长录音来减轻某些言语任务（如独白）中性能下降的问题。我们的研究结果表明，性能和可解释性并不一定是互斥的。未来的工作将增强非专业人士使用解释的便利性，并探索严重程度估计以提高实际临床相关性。 

---
# Causal-SAM-LLM: Large Language Models as Causal Reasoners for Robust Medical Segmentation 

**Title (ZH)**: 因果-SAM-LLM：大语言模型作为因果推理器以实现稳健的医学分割 

**Authors**: Tao Tang, Shijie Xu, Yiting Wu, Zhixiang Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.03585)  

**Abstract**: The clinical utility of deep learning models for medical image segmentation is severely constrained by their inability to generalize to unseen domains. This failure is often rooted in the models learning spurious correlations between anatomical content and domain-specific imaging styles. To overcome this fundamental challenge, we introduce Causal-SAM-LLM, a novel framework that elevates Large Language Models (LLMs) to the role of causal reasoners. Our framework, built upon a frozen Segment Anything Model (SAM) encoder, incorporates two synergistic innovations. First, Linguistic Adversarial Disentanglement (LAD) employs a Vision-Language Model to generate rich, textual descriptions of confounding image styles. By training the segmentation model's features to be contrastively dissimilar to these style descriptions, it learns a representation robustly purged of non-causal information. Second, Test-Time Causal Intervention (TCI) provides an interactive mechanism where an LLM interprets a clinician's natural language command to modulate the segmentation decoder's features in real-time, enabling targeted error correction. We conduct an extensive empirical evaluation on a composite benchmark from four public datasets (BTCV, CHAOS, AMOS, BraTS), assessing generalization under cross-scanner, cross-modality, and cross-anatomy settings. Causal-SAM-LLM establishes a new state of the art in out-of-distribution (OOD) robustness, improving the average Dice score by up to 6.2 points and reducing the Hausdorff Distance by 15.8 mm over the strongest baseline, all while using less than 9% of the full model's trainable parameters. Our work charts a new course for building robust, efficient, and interactively controllable medical AI systems. 

**Abstract (ZH)**: 深度学习模型在医学图像分割中的临床应用受到其难以泛化到未见领域的能力限制。这一失败往往源于模型学习了解剖内容与特定成像风格间的虚假关联。为克服这一根本性挑战，我们引入了因果-SAM-LLM（Causal-SAM-LLM）这一新颖框架，将大型语言模型（LLM）提升为因果推理者的角色。基于冻结的Segment Anything Model（SAM）编码器，我们的框架集成了两项协同创新。首先，语言对抗脱噪（LAD）利用视觉-语言模型生成丰富、文本化的混杂图像风格描述，并通过训练分割模型的特征与这些风格描述形成对比性差异，从而学习到不受非因果信息污染的表示。其次，测试时因果干预（TCI）提供了一个交互机制，其中LLM通过解释临床人员的自然语言指令实时调整分割解码器的特征，实现精准的错误修正。我们对来自四个公开数据集（BTCV、CHAOS、AMOS、BraTS）的综合基准进行了广泛的经验评估，考察了跨扫描仪、跨模态和跨器官设置下的泛化能力。Causal-SAM-LLM 在分布外（OOD）鲁棒性上建立了新的标准，平均狄氏分数提高了6.2个百分点，并将Hausdorff距离减少了15.8毫米，同时仅使用了全模型不到9%的可训练参数。本项工作为构建稳健、高效且可交互控制的医疗AI系统开辟了新的路径。 

---
# SciVid: Cross-Domain Evaluation of Video Models in Scientific Applications 

**Title (ZH)**: SciVid: 在科学应用中跨领域评估视频模型 

**Authors**: Yana Hasson, Pauline Luc, Liliane Momeni, Maks Ovsjanikov, Guillaume Le Moing, Alina Kuznetsova, Ira Ktena, Jennifer J. Sun, Skanda Koppula, Dilara Gokay, Joseph Heyward, Etienne Pot, Andrew Zisserman  

**Link**: [PDF](https://arxiv.org/pdf/2507.03578)  

**Abstract**: In recent years, there has been a proliferation of spatiotemporal foundation models in different scientific disciplines. While promising, these models are often domain-specific and are only assessed within the particular applications for which they are designed. Given that many tasks can be represented as video modeling problems, video foundation models (ViFMs) hold considerable promise as general-purpose domain-agnostic approaches. However, it is not known whether the knowledge acquired on large-scale but potentially out-of-domain data can be effectively transferred across diverse scientific disciplines, and if a single, pretrained ViFM can be competitive with domain-specific baselines. To address this, we introduce SciVid, a comprehensive benchmark comprising five *Sci*entific *Vid*eo tasks, across medical computer vision, animal behavior, and weather forecasting. We adapt six leading ViFMs to SciVid using simple trainable readout modules, establishing strong baselines and demonstrating the potential for effective transfer learning. Specifically, we show that state-of-the-art results can be obtained in several applications by leveraging the general-purpose representations from ViFM backbones. Furthermore, our results reveal the limitations of existing ViFMs, and highlight opportunities for the development of generalizable models for high-impact scientific applications. We release our code at this https URL to facilitate further research in the development of ViFMs. 

**Abstract (ZH)**: 近年来，不同科学学科中涌现出了大量的时空基础模型。尽管前景广阔，但这些模型通常具有领域特异性，并仅在所设计的应用场景中进行评估。鉴于许多任务可以表示为视频建模问题，视频基础模型(ViFMs)作为通用领域无关的方法显示出巨大潜力。然而，尚不清楚在大规模但可能是域外的数据上获得的知识能否有效转移到不同科学学科中，以及是否可以使用预训练的ViFM与领域特定基准相竞争。为解决这一问题，我们引入了SciVid，这是一个全面的基准，包含了五大科学视频任务，涵盖医学计算机视觉、动物行为和天气预报等领域。我们使用简单的可训练读出模块将六个领先的ViFMs适应到SciVid上，建立了强大的基线，并展示了有效的迁移学习潜力。具体而言，我们展示了可以通过利用ViFM主干的一般表示在多种应用中获得最先进的结果。此外，我们的结果揭示了现有ViFMs的局限性，并强调了为高影响力科学应用开发可迁移模型的机会。我们已在以下链接发布代码，以促进ViFMs开发的研究：[请提供实际的链接] 

---
# An Advanced Deep Learning Framework for Ischemic and Hemorrhagic Brain Stroke Diagnosis Using Computed Tomography (CT) Images 

**Title (ZH)**: 基于计算机断层扫描（CT）图像的缺血性和出血性脑卒中诊断的高级深度学习框架 

**Authors**: Md. Sabbir Hossen, Eshat Ahmed Shuvo, Shibbir Ahmed Arif, Pabon Shaha, Md. Saiduzzaman, Mostofa Kamal Nasir  

**Link**: [PDF](https://arxiv.org/pdf/2507.03558)  

**Abstract**: Brain stroke is one of the leading causes of mortality and long-term disability worldwide, highlighting the need for precise and fast prediction techniques. Computed Tomography (CT) scan is considered one of the most effective methods for diagnosing brain strokes. The majority of stroke classification techniques rely on a single slice-level prediction mechanism, allowing the radiologist to manually choose the most critical CT slice from the original CT volume. Although clinical evaluations are often used in traditional diagnostic procedures, machine learning (ML) has opened up new avenues for improving stroke diagnosis. To supplement traditional diagnostic techniques, this study investigates the use of machine learning models, specifically concerning the prediction of brain stroke at an early stage utilizing CT scan images. In this research, we proposed a novel approach to brain stroke detection leveraging machine learning techniques, focusing on optimizing classification performance with pre-trained deep learning models and advanced optimization strategies. Pre-trained models, including DenseNet201, InceptionV3, MobileNetV2, ResNet50, and Xception, are utilized for feature extraction. Additionally, we employed feature engineering techniques, including BFO, PCA, and LDA, to enhance models' performance further. These features are subsequently classified using machine learning algorithms such as SVC, RF, XGB, DT, LR, KNN, and GNB. Our experiments demonstrate that the combination of MobileNetV2, LDA, and SVC achieved the highest classification accuracy of 97.93%, significantly outperforming other model-optimizer-classifier combinations. The results underline the effectiveness of integrating lightweight pre-trained models with robust optimization and classification techniques for brain stroke diagnosis. 

**Abstract (ZH)**: 全球范围内，脑卒中是导致死亡和长期残疾的主要原因之一，突显了精确和快速预测技术的必要性。CT扫描被认为是诊断脑卒中最有效的手段之一。大多数脑卒中分类技术依赖于单个切片级别的预测机制，允许放射学家从原始CT体积中手动选择最关键的CT切片。尽管传统的诊疗流程中通常会使用临床评估，但机器学习（ML）已开辟了提高脑卒中诊断的新途径。为了补充传统的诊断技术，本研究探讨了机器学习模型的应用，特别是在CT扫描图像上早期预测脑卒中的方法。在本研究中，我们提出了利用机器学习技术进行脑卒中检测的新型方法，专注于使用预训练深度学习模型和高级优化策略优化分类性能。我们使用了包括DenseNet201、InceptionV3、MobileNetV2、ResNet50和Xception在内的预训练模型进行特征提取，并运用包括BFO、PCA和LDA在内的特征工程技术进一步提升模型性能。随后使用SVC、RF、XGB、DT、LR、KNN和GNB等机器学习算法对这些特征进行分类。我们的实验表明，MobileNetV2、LDA和SVC的组合实现了最高的分类准确率97.93%，明显优于其他模型-优化器-分类器组合。结果表明，将轻量级预训练模型与稳健的优化和分类技术结合使用对于脑卒中诊断的有效性。 

---
# H2HTalk: Evaluating Large Language Models as Emotional Companion 

**Title (ZH)**: H2HTalk: 评估大型语言模型作为情感伴侣的有效性 

**Authors**: Boyang Wang, Yalun Wu, Hongcheng Guo, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.03543)  

**Abstract**: As digital emotional support needs grow, Large Language Model companions offer promising authentic, always-available empathy, though rigorous evaluation lags behind model advancement. We present Heart-to-Heart Talk (H2HTalk), a benchmark assessing companions across personality development and empathetic interaction, balancing emotional intelligence with linguistic fluency. H2HTalk features 4,650 curated scenarios spanning dialogue, recollection, and itinerary planning that mirror real-world support conversations, substantially exceeding previous datasets in scale and diversity. We incorporate a Secure Attachment Persona (SAP) module implementing attachment-theory principles for safer interactions. Benchmarking 50 LLMs with our unified protocol reveals that long-horizon planning and memory retention remain key challenges, with models struggling when user needs are implicit or evolve mid-conversation. H2HTalk establishes the first comprehensive benchmark for emotionally intelligent companions. We release all materials to advance development of LLMs capable of providing meaningful and safe psychological support. 

**Abstract (ZH)**: 数字情感支持需求增长之际，大型语言模型伴侣提供有希望的真实且始终可用的同理心，尽管严格的评估滞后于模型进步。我们呈现心灵至心灵交谈（H2HTalk）基准测试，该基准测试评估伴侣在个性发展和同理互动方面的表现，平衡情感智能与语言流畅性。H2HTalk包含4,650个精心策划的场景，涵盖对话、回忆和行程规划，反映真实世界的支持对话，显著超越了以往数据集在规模和多样性上的限制。我们引入一个安全依附人格（SAP）模块，实施依附理论原则，以确保更安全的交互。使用统一协议对50个LLM进行基准测试表明，长期规划和记忆保留仍然是关键挑战，模型在用户需求含蓄或在对话中途变化时表现挣扎。H2HTalk建立了首个全面的情感智能伴侣基准测试。我们发布所有材料以促进能够提供有意义且安全的心理支持的LLM的发展。 

---
# Foundation versus Domain-specific Models: Performance Comparison, Fusion, and Explainability in Face Recognition 

**Title (ZH)**: 基础模型与领域特定模型：面部识别性能比较、融合及解释性 

**Authors**: Redwan Sony, Parisa Farmanifard, Arun Ross, Anil K. Jain  

**Link**: [PDF](https://arxiv.org/pdf/2507.03541)  

**Abstract**: In this paper, we address the following question: How do generic foundation models (e.g., CLIP, BLIP, LLaVa, DINO) compare against a domain-specific face recognition model (viz., AdaFace or ArcFace) on the face recognition task? Through a series of experiments involving several foundation models and benchmark datasets, we are able to report the following findings: (a) In all datasets considered, domain-specific models outperformed zero-shot foundation models. (b) The performance of zero-shot generic foundation models improves on over-segmented face images than tightly cropped faces thereby suggesting the importance of contextual clues. For example, at a False Match Rate (FMR) of 0.01%, the True Match Rate (TMR) of OpenCLIP improved from 64.97% to 81.73% on the LFW dataset as the face crop increased from 112x112 to 250x250 while the TMR of domain-specific AdaFace dropped from 99.09% to 77.31%. (c) A simple score-level fusion of a foundation model with a domain-specific FR model improved the accuracy at low FMRs. For example, the TMR of AdaFace when fused with BLIP improved from 72.64% to 83.31% at an FMR of 0.0001% on the IJB-B dataset and from 73.17% to 85.81% on the IJB-C dataset. (d) Foundation models, such as ChatGPT, can be used to impart explainability to the FR pipeline (e.g., ``Despite minor lighting and head tilt differences, the two left-profile images show high consistency in forehead slope, nose shape, chin contour...''). In some instances, foundation models are even able to resolve low-confidence decisions made by AdaFace (e.g., ``Although AdaFace assigns a low similarity score of 0.21, both images exhibit visual similarity...and the pair is likely of the same person''), thereby reiterating the importance of combining domain-specific FR models with generic foundation models in a judicious manner. 

**Abstract (ZH)**: 基于基础模型与领域特定面部识别模型在面部识别任务中的比较研究 

---
# Multimodal Alignment with Cross-Attentive GRUs for Fine-Grained Video Understanding 

**Title (ZH)**: 多模态注意力GRUs的细粒度视频理解 

**Authors**: Namho Kim, Junhwa Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.03531)  

**Abstract**: Fine-grained video classification requires understanding complex spatio-temporal and semantic cues that often exceed the capacity of a single modality. In this paper, we propose a multimodal framework that fuses video, image, and text representations using GRU-based sequence encoders and cross-modal attention mechanisms. The model is trained using a combination of classification or regression loss, depending on the task, and is further regularized through feature-level augmentation and autoencoding techniques. To evaluate the generality of our framework, we conduct experiments on two challenging benchmarks: the DVD dataset for real-world violence detection and the Aff-Wild2 dataset for valence-arousal estimation. Our results demonstrate that the proposed fusion strategy significantly outperforms unimodal baselines, with cross-attention and feature augmentation contributing notably to robustness and performance. 

**Abstract (ZH)**: 细粒度视频分类需要理解复杂的时空和语义线索，这些线索通常超出了单一模态的能力。本文提出了一种多模态框架，该框架使用基于GRU的序列编码器和跨模态注意机制融合视频、图像和文本表示。该模型根据不同任务使用分类或回归损失进行训练，并通过特征层面的增强和自编码技术进一步正则化。为了评估我们框架的一般性，我们在两个具有挑战性的基准上进行了实验：用于真实世界暴力检测的DVD数据集和用于情感估值的Aff-Wild2数据集。实验结果表明，提出的融合策略显著优于单模态基线，跨注意和特征增强对鲁棒性和性能的提升尤为显著。 

---
# Generating Synthetic Relational Tabular Data via Structural Causal Models 

**Title (ZH)**: 通过结构因果模型生成合成关系型表格数据 

**Authors**: Frederik Hoppe, Astrid Franz, Lars Kleinemeier, Udo Göbel  

**Link**: [PDF](https://arxiv.org/pdf/2507.03528)  

**Abstract**: Synthetic tabular data generation has received increasing attention in recent years, particularly with the emergence of foundation models for tabular data. The breakthrough success of TabPFN (Hollmann et al.,2025), which leverages vast quantities of synthetic tabular datasets derived from structural causal models (SCMs), demonstrates the critical role synthetic data plays in developing powerful tabular foundation models. However, most real-world tabular data exists in relational formats spanning multiple interconnected tables - a structure not adequately addressed by current generation methods. In this work, we extend the SCM-based approach by developing a novel framework that generates realistic synthetic relational tabular data including causal relationships across tables. Our experiments confirm that this framework is able to construct relational datasets with complex inter-table dependencies mimicking real-world scenarios. 

**Abstract (ZH)**: 合成表格数据生成在最近几年受到了越来越多的关注，尤其是在面向表格数据的基础模型出现之后。基于结构因果模型（SCMs）生成的大规模合成表格数据（Hollmann et al., 2025）的突破性成功，展示了合成数据在开发强大表格基础模型中的关键作用。然而，大多数实际的表格数据以跨越多个相互关联表格的关联格式存在——这种结构目前的生成方法尚未充分解决。在本工作中，我们通过开发一种新的框架扩展了基于SCM的方法，该框架能够生成包含表格间因果关系的现实合成关联表格数据。我们的实验证实，该框架能够构建具有复杂跨表格依赖关系的关联数据集，以模拟现实世界场景。 

---
# Reinforcement Learning-based Feature Generation Algorithm for Scientific Data 

**Title (ZH)**: 基于强化学习的科学数据特征生成算法 

**Authors**: Meng Xiao, Junfeng Zhou, Yuanchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.03498)  

**Abstract**: Feature generation (FG) aims to enhance the prediction potential of original data by constructing high-order feature combinations and removing redundant features. It is a key preprocessing step for tabular scientific data to improve downstream machine-learning model performance. Traditional methods face the following two challenges when dealing with the feature generation of scientific data: First, the effective construction of high-order feature combinations in scientific data necessitates profound and extensive domain-specific expertise. Secondly, as the order of feature combinations increases, the search space expands exponentially, imposing prohibitive human labor consumption. Advancements in the Data-Centric Artificial Intelligence (DCAI) paradigm have opened novel avenues for automating feature generation processes. Inspired by that, this paper revisits the conventional feature generation workflow and proposes the Multi-agent Feature Generation (MAFG) framework. Specifically, in the iterative exploration stage, multi-agents will construct mathematical transformation equations collaboratively, synthesize and identify feature combinations ex-hibiting high information content, and leverage a reinforcement learning mechanism to evolve their strategies. Upon completing the exploration phase, MAFG integrates the large language models (LLMs) to interpreta-tively evaluate the generated features of each significant model performance breakthrough. Experimental results and case studies consistently demonstrate that the MAFG framework effectively automates the feature generation process and significantly enhances various downstream scientific data mining tasks. 

**Abstract (ZH)**: 特征生成（FG）旨在通过构建高阶特征组合和去除冗余特征来增强原始数据的预测潜力。它是提高表格式科学数据下游机器学习模型性能的关键预处理步骤。传统方法在处理科学数据的特征生成时面临以下两个挑战：首先，有效构建科学数据中的高阶特征组合需要深厚广泛的专业领域知识。其次，随着特征组合阶数的增加，搜索空间会呈指数级扩大，导致巨大的人力劳动消耗。数据为中心的人工智能（DCAI）范式的进步为自动化特征生成过程开辟了新的途径。受此启发，本文重新审视了传统的特征生成工作流，并提出了多代理特征生成（MAFG）框架。特别地，在迭代探索阶段，多代理将协作构建数学变换方程，综合并识别信息含量高的特征组合，并利用强化学习机制演化其策略。在完成探索阶段后，MAFG结合大型语言模型（LLMs）进行解释性评价，以评估每个重大模型性能突破中生成的特征。实验结果和案例研究一致地证明，MAFG框架有效地自动化了特征生成过程，并显著增强了各种下游科学数据挖掘任务。 

---
# BMMR: A Large-Scale Bilingual Multimodal Multi-Discipline Reasoning Dataset 

**Title (ZH)**: BMMR：大规模跨模态多学科双语推理数据集 

**Authors**: Zhiheng Xi, Guanyu Li, Yutao Fan, Honglin Guo, Yufang Liu, Xiaoran Fan, Jiaqi Liu, Jingchao Ding, Wangmeng Zuo, Zhenfei Yin, Lei Bai, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03483)  

**Abstract**: In this paper, we introduce BMMR, a large-scale bilingual, multimodal, multi-disciplinary reasoning dataset for the community to develop and evaluate large multimodal models (LMMs). BMMR comprises 110k college-level questions spanning 300 UNESCO-defined subjects, spanning diverse formats-multiple-choice, fill-in-the-blank, and open-ended QA-and sourced from both print and digital media such as books, exams, and quizzes. All data are curated and filtered via a human-in-the-loop and scalable framework, and each instance is paired with a high-quality reasoning path. The dataset is organized into two parts: BMMR-Eval that comprises 20,458 high-quality instances to comprehensively assess LMMs' knowledge and reasoning across multiple disciplines in both Chinese and English; and BMMR-Train that contains 88,991 instances to support further research and development, extending the current focus on mathematical reasoning to diverse disciplines and domains. In addition, we propose the process-based multi-discipline verifier (i.e., BMMR-Verifier) for accurate and fine-grained evaluation of reasoning paths. Extensive experiments on 24 models reveal that (i) even SOTA models (e.g., o3 and Gemini-2.5-Pro) leave substantial headroom on BMMR-Eval; (ii) reasoning models exhibit discipline bias and outperform LMMs only on specific subjects; (iii) open-source models still trail their proprietary counterparts; and (iv) fine-tuning on BMMR-Train narrows this gap. Additionally, we conduct reasoning-chain analyses using BMMR-Verifier and other in-depth studies, uncovering the challenges LMMs currently face in multidisciplinary reasoning. We will release the data, and we hope our work can offer insights and contributions to the community. 

**Abstract (ZH)**: 基于BMMR的大规模跨模态多学科推理数据集及其评估方法 

---
# Beyond Weaponization: NLP Security for Medium and Lower-Resourced Languages in Their Own Right 

**Title (ZH)**: 超越武器化：中低资源语言的自身权利下的自然语言处理安全 

**Authors**: Heather Lent  

**Link**: [PDF](https://arxiv.org/pdf/2507.03473)  

**Abstract**: Despite mounting evidence that multilinguality can be easily weaponized against language models (LMs), works across NLP Security remain overwhelmingly English-centric. In terms of securing LMs, the NLP norm of "English first" collides with standard procedure in cybersecurity, whereby practitioners are expected to anticipate and prepare for worst-case outcomes. To mitigate worst-case outcomes in NLP Security, researchers must be willing to engage with the weakest links in LM security: lower-resourced languages. Accordingly, this work examines the security of LMs for lower- and medium-resourced languages. We extend existing adversarial attacks for up to 70 languages to evaluate the security of monolingual and multilingual LMs for these languages. Through our analysis, we find that monolingual models are often too small in total number of parameters to ensure sound security, and that while multilinguality is helpful, it does not always guarantee improved security either. Ultimately, these findings highlight important considerations for more secure deployment of LMs, for communities of lower-resourced languages. 

**Abstract (ZH)**: 尽管有大量的证据表明多语言能力可以被轻松武器化用于语言模型（LMs），NLP安全领域的研究仍然主要集中在英语上。为了在NLP安全中减轻最坏情况的结果，研究人员必须愿意关注LM安全中最薄弱的环节：低资源语言。因此，本工作研究了低资源和中资源语言的LM安全性。我们将现有的对抗性攻击扩展到多达70种语言，以评估这些语言的单语言和多语言LM的安全性。通过对这些语言的LM进行分析，我们发现单语言模型往往由于参数总量较少而难以确保安全性，尽管多语言能力有帮助，但它也不总是能够保证安全性提升。最终，这些发现强调了更安全地部署LMs对于低资源语言社区的重要考虑。 

---
# Helping CLIP See Both the Forest and the Trees: A Decomposition and Description Approach 

**Title (ZH)**: 帮助CLIP同时看到森林和树木：一种分解与描述的方法 

**Authors**: Leyan Xue, Zongbo Han, Guangyu Wang, Qinghua Hu, Mingyue Cheng, Changqing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03458)  

**Abstract**: Vision-Language Models (VLMs) like CLIP achieve cross-modal semantic alignment through contrastive learning, exhibiting robust zero-shot generalization. Traditional prompt engineering, however, predominantly relies on coarse-grained category labels, neglecting fine-grained local semantics. Existing approaches assume that VLMs inherently recognize localized visual details and attempt to enhance classification by augmenting text prompts with attribute descriptors generated by large language models. However, our systematic experiments reveal critical limitations: CLIP's strong bias toward global image patterns hinders its ability to process localized visual descriptors. To address this fundamental constraint, we propose a simple, effective, and plug-and-play solution that enables CLIP to ``See Both the Forest and the Trees." Specifically, we employ stochastic multi-crop augmentation to activate CLIP's latent capacity for localized feature analysis. By cropping only partial regions, the approach effectively constrains the model's receptive field and recalibrates its attention mechanism, thereby mitigating its inherent bias. We evaluate the proposed method under zero-shot, few-shot, and test-time adaptation settings, and extensive experiments demonstrate that D&D achieves promising performance. 

**Abstract (ZH)**: 基于视觉-语言模型（VLMs）如CLIP通过对比学习实现跨模态语义对齐，并表现出鲁棒的零样本泛化能力。然而，传统的提示工程技术主要依赖于粗粒度类别标签，忽视了细粒度局部语义。现有方法假设VLMs能够自然识别局部视觉细节，并尝试通过使用大型语言模型生成的属性描述符增强文本提示以提高分类性能。然而，我们的系统性实验揭示了关键限制：CLIP对全局图像模式的强烈偏向限制了其处理局部视觉描述符的能力。为解决这一根本性约束，我们提出了一种简单、有效且即插即用的解决方案，使CLIP能够“见微知著”。具体而言，我们采用随机多裁剪增强方法激活CLIP在局部特征分析方面的潜在能力。通过仅裁剪部分区域，该方法有效地限制了模型的感受野，并重新校准了其注意力机制，从而减轻了其固有的偏见。我们在零样本、少样本和测试时适应设置下评估了所提出的方法，并进行的大量实验表明，D&D取得了令人鼓舞的性能。 

---
# Evaluating the Evaluators: Trust in Adversarial Robustness Tests 

**Title (ZH)**: 评估评估者： adversarial 稳定性测试中的信任 

**Authors**: Antonio Emanuele Cinà, Maura Pintor, Luca Demetrio, Ambra Demontis, Battista Biggio, Fabio Roli  

**Link**: [PDF](https://arxiv.org/pdf/2507.03450)  

**Abstract**: Despite significant progress in designing powerful adversarial evasion attacks for robustness verification, the evaluation of these methods often remains inconsistent and unreliable. Many assessments rely on mismatched models, unverified implementations, and uneven computational budgets, which can lead to biased results and a false sense of security. Consequently, robustness claims built on such flawed testing protocols may be misleading and give a false sense of security. As a concrete step toward improving evaluation reliability, we present AttackBench, a benchmark framework developed to assess the effectiveness of gradient-based attacks under standardized and reproducible conditions. AttackBench serves as an evaluation tool that ranks existing attack implementations based on a novel optimality metric, which enables researchers and practitioners to identify the most reliable and effective attack for use in subsequent robustness evaluations. The framework enforces consistent testing conditions and enables continuous updates, making it a reliable foundation for robustness verification. 

**Abstract (ZH)**: 尽管在设计强大的对抗性规避攻击以进行鲁棒性验证方面取得了显著进展，但这些方法的评估往往仍然不一致且不可靠。许多评估依赖于不匹配的模型、未经验证的实现以及不均衡的计算预算，这可能会导致偏颇的结果和虚假的安全感。因此，基于此类有缺陷的测试协议提出的鲁棒性声明可能是误导性的，并会给人一种虚假的安全感。为进一步提高评估的可靠性，我们提出了一种名为AttackBench的基准框架，用于在标准化和可重现的条件下评估基于梯度的攻击的有效性。AttackBench充当了一个评估工具，根据新型最优性度量对现有的攻击实现进行排名，使研究人员和实践者能够识别出在后续鲁棒性评估中最为可靠和有效的攻击。该框架确保了测试条件的一致性，并允许持续更新，使其成为鲁棒性验证的可靠基础。 

---
# Improving Social Determinants of Health Documentation in French EHRs Using Large Language Models 

**Title (ZH)**: 使用大型语言模型改善法语电子健康记录中的社会决定因素记录 

**Authors**: Adrien Bazoge, Pacôme Constant dit Beaufils, Mohammed Hmitouch, Romain Bourcier, Emmanuel Morin, Richard Dufour, Béatrice Daille, Pierre-Antoine Gourraud, Matilde Karakachoff  

**Link**: [PDF](https://arxiv.org/pdf/2507.03433)  

**Abstract**: Social determinants of health (SDoH) significantly influence health outcomes, shaping disease progression, treatment adherence, and health disparities. However, their documentation in structured electronic health records (EHRs) is often incomplete or missing. This study presents an approach based on large language models (LLMs) for extracting 13 SDoH categories from French clinical notes. We trained Flan-T5-Large on annotated social history sections from clinical notes at Nantes University Hospital, France. We evaluated the model at two levels: (i) identification of SDoH categories and associated values, and (ii) extraction of detailed SDoH with associated temporal and quantitative information. The model performance was assessed across four datasets, including two that we publicly release as open resources. The model achieved strong performance for identifying well-documented categories such as living condition, marital status, descendants, job, tobacco, and alcohol use (F1 score > 0.80). Performance was lower for categories with limited training data or highly variable expressions, such as employment status, housing, physical activity, income, and education. Our model identified 95.8% of patients with at least one SDoH, compared to 2.8% for ICD-10 codes from structured EHR data. Our error analysis showed that performance limitations were linked to annotation inconsistencies, reliance on English-centric tokenizer, and reduced generalizability due to the model being trained on social history sections only. These results demonstrate the effectiveness of NLP in improving the completeness of real-world SDoH data in a non-English EHR system. 

**Abstract (ZH)**: 社会决定因素对健康的影響（SDoH）顯著影響健康結果，塑造疾病的進展、治療依從性和健康不平等。然而，這些因素在結構化的電子健康紀錄（EHRs）中的記錄往往不完整或缺失。本研究提出了一種基於大型語言模型（LLMs）的方法，用於從法國臨床記錄中提取13類SDoH。我們在法國南特大學醫院的臨床記錄中标註的社會史部分上訓練了Flan-T5-Large。我們在兩個層面上評估了模型的表現：（i）識別SDoH類別及其相關值，以及（ii）提取帶有關聯時間和量化信息的詳細SDoH。我們在四個數據集中評估了模型的表現，包括兩個我們公開釋出作為開源資源的數據集。模型在生活條件、婚姻狀態、後代、職業、 tobacco 和酒精使用等 хорошо文書化類別的識別上表現出色（F1分數>0.80）。但在訓練數據有限或表達高度多變的類別，如就業狀態、住房、體育活動、收入和教育方面，表現較低。本研究模型識別出至少一項SDoH的患者佔95.8%，而基于結構化EHR數據的ICD-10碼僅為2.8%。我們的錯誤分析表明，表現限制與標注不一致、依賴英語为中心的分词器以及模型僅訓練於社會史部分而导致的泛化能力不足有關。這些結果展示了NLP在改善非英語EHR系統中真實世界SDoH數據完整性方面的有效性。 

---
# Multi-Level Fusion Graph Neural Network for Molecule Property Prediction 

**Title (ZH)**: 多层融合图神经网络在分子性质预测中的应用 

**Authors**: XiaYu Liu, Hou-biao Li, Yang Liu, Chao Fan  

**Link**: [PDF](https://arxiv.org/pdf/2507.03430)  

**Abstract**: Accurate molecular property prediction is essential in drug discovery and related fields. However, existing graph neural networks (GNNs) often struggle to simultaneously capture both local and global molecular structures. In this work, we propose a Multi-Level Fusion Graph Neural Network (MLFGNN) that integrates Graph Attention Networks and a novel Graph Transformer to jointly model local and global dependencies. In addition, we incorporate molecular fingerprints as a complementary modality and introduce a mechanism of interaction between attention to adaptively fuse information across representations. Extensive experiments on multiple benchmark datasets demonstrate that MLFGNN consistently outperforms state-of-the-art methods in both classification and regression tasks. Interpretability analysis further reveals that the model effectively captures task-relevant chemical patterns, supporting the usefulness of multi-level and multi-modal fusion in molecular representation learning. 

**Abstract (ZH)**: 多级融合图神经网络在药物发现相关领域的分子性质预测中至关重要：MLFGNN在局部和全局依赖性的联合建模中取得了优越性能。 

---
# Pose-Star: Anatomy-Aware Editing for Open-World Fashion Images 

**Title (ZH)**: Pose-Star: 具有解剖意识的开放世界服装图像编辑 

**Authors**: Yuran Dong, Mang Ye  

**Link**: [PDF](https://arxiv.org/pdf/2507.03402)  

**Abstract**: To advance real-world fashion image editing, we analyze existing two-stage pipelines(mask generation followed by diffusion-based editing)which overly prioritize generator optimization while neglecting mask controllability. This results in two critical limitations: I) poor user-defined flexibility (coarse-grained human masks restrict edits to predefined regions like upper torso; fine-grained clothes masks preserve poses but forbid style/length customization). II) weak pose robustness (mask generators fail due to articulated poses and miss rare regions like waist, while human parsers remain limited by predefined categories). To address these gaps, we propose Pose-Star, a framework that dynamically recomposes body structures (e.g., neck, chest, etc.) into anatomy-aware masks (e.g., chest-length) for user-defined edits. In Pose-Star, we calibrate diffusion-derived attention (Star tokens) via skeletal keypoints to enhance rare structure localization in complex poses, suppress noise through phase-aware analysis of attention dynamics (Convergence,Stabilization,Divergence) with threshold masking and sliding-window fusion, and refine edges via cross-self attention merging and Canny alignment. This work bridges controlled benchmarks and open-world demands, pioneering anatomy-aware, pose-robust editing and laying the foundation for industrial fashion image editing. 

**Abstract (ZH)**: 为进一步推动现实世界中的时尚图像编辑，我们分析了现有的两阶段流水线（掩码生成后进行基于扩散的编辑），这些流水线过度注重生成器优化而忽视了掩码的可控性。这导致了两个关键限制：I）较差的用户定义灵活性（粗粒度的人体掩码限制了编辑范围，仅限于预定义区域如上半身；细粒度的服饰掩码保留姿势但禁止风格和长度的定制）。II）较弱的姿态鲁棒性（掩码生成器在复杂姿态下失败，未能捕捉到腰部等稀有区域，而人体解析器仍受限于预定义类别）。为解决这些问题，我们提出了一种名为Pose-Star的框架，该框架动态重组身体结构（如颈部、胸部等）以形成解剖学意识的掩码（如胸部长度），用于用户定义的编辑。在Pose-Star中，我们通过骨骼关键点校准扩散引导的注意力（Star令牌），以增强复杂姿态中稀有结构的定位，通过相位感知的注意力动态分析（收敛、稳定、发散）结合阈值掩蔽和滑动窗口融合来抑制噪声，并通过交叉自我注意力合并和Canny对齐来细化边缘。本工作连接了可控基准和开放世界的实际需求，开创了解剖学意识、姿态鲁棒的编辑方法，并为工业时尚图像编辑奠定了基础。 

---
# LLM4Hint: Leveraging Large Language Models for Hint Recommendation in Offline Query Optimization 

**Title (ZH)**: LLM4Hint：利用大型语言模型进行离线查询优化中的提示推荐 

**Authors**: Suchen Liu, Jun Gao, Yinjun Han, Yang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.03384)  

**Abstract**: Query optimization is essential for efficient SQL query execution in DBMS, and remains attractive over time due to the growth of data volumes and advances in hardware. Existing traditional optimizers struggle with the cumbersome hand-tuning required for complex workloads, and the learning-based methods face limitations in ensuring generalization. With the great success of Large Language Model (LLM) across diverse downstream tasks, this paper explores how LLMs can be incorporated to enhance the generalization of learned optimizers. Though promising, such an incorporation still presents challenges, mainly including high model inference latency, and the substantial fine-tuning cost and suboptimal performance due to inherent discrepancy between the token sequences in LLM and structured SQL execution plans with rich numerical features.
In this paper, we focus on recurring queries in offline optimization to alleviate the issue of high inference latency, and propose \textbf{LLM4Hint} that leverages moderate-sized backbone LLMs to recommend query optimization hints. LLM4Hint achieves the goals through: (i) integrating a lightweight model to produce a soft prompt, which captures the data distribution in DBMS and the SQL predicates to provide sufficient optimization features while simultaneously reducing the context length fed to the LLM, (ii) devising a query rewriting strategy using a larger commercial LLM, so as to simplify SQL semantics for the backbone LLM and reduce fine-tuning costs, and (iii) introducing an explicit matching prompt to facilitate alignment between the LLM and the lightweight model, which can accelerate convergence of the combined model. Experiments show that LLM4Hint, by leveraging the LLM's stronger capability to understand the query statement, can outperform the state-of-the-art learned optimizers in terms of both effectiveness and generalization. 

**Abstract (ZH)**: 基于大语言模型的查询优化提示（LLM4Hint） 

---
# Be the Change You Want to See: Revisiting Remote Sensing Change Detection Practices 

**Title (ZH)**: 欲成所愿之变：重访遥感变化检测实践 

**Authors**: Blaž Rolih, Matic Fučka, Filip Wolf, Luka Čehovin Zajc  

**Link**: [PDF](https://arxiv.org/pdf/2507.03367)  

**Abstract**: Remote sensing change detection aims to localize semantic changes between images of the same location captured at different times. In the past few years, newer methods have attributed enhanced performance to the additions of new and complex components to existing architectures. Most fail to measure the performance contribution of fundamental design choices such as backbone selection, pre-training strategies, and training configurations. We claim that such fundamental design choices often improve performance even more significantly than the addition of new architectural components. Due to that, we systematically revisit the design space of change detection models and analyse the full potential of a well-optimised baseline. We identify a set of fundamental design choices that benefit both new and existing architectures. Leveraging this insight, we demonstrate that when carefully designed, even an architecturally simple model can match or surpass state-of-the-art performance on six challenging change detection datasets. Our best practices generalise beyond our architecture and also offer performance improvements when applied to related methods, indicating that the space of fundamental design choices has been underexplored. Our guidelines and architecture provide a strong foundation for future methods, emphasizing that optimizing core components is just as important as architectural novelty in advancing change detection performance. Code: this https URL 

**Abstract (ZH)**: 遥感变化检测旨在 localization 同一位置在不同时段拍摄的图像之间的语义变化。在过去几年中，新的方法通过向现有架构添加新而复杂的组件来提高性能。大多数方法未能衡量基础设计选择（如主干网络选择、预训练策略和训练配置）对性能的贡献。我们认为，这些基础设计选择往往比添加新架构组件对性能的提升更为显著。因此，我们系统地重新审视了变化检测模型的设计空间，并分析了优化基准模型的全部潜力。我们确定了一组对新旧架构均有益的基础设计选择。利用这一洞察，我们证明，当精心设计时，即使一个架构简单的模型也能在六个具有挑战性的变化检测数据集中达到或超越最先进的性能。我们的最佳实践超越了我们的架构，并且当应用于相关方法时也能提供性能改进，这表明基础设计选择的空间尚未充分利用。我们的指南和架构为未来的方法提供了一个坚实的基础，强调优化核心组件与架构新颖性同样重要，以推动变化检测性能的提升。代码：this https URL 

---
# Backtesting Sentiment Signals for Trading: Evaluating the Viability of Alpha Generation from Sentiment Analysis 

**Title (ZH)**: 基于情绪信号的交易回测：情绪分析Alpha生成可行性的评估 

**Authors**: Elvys Linhares Pontes, Carlos-Emiliano González-Gallardo, Georgeta Bordea, José G. Moreno, Mohamed Ben Jannet, Yuxuan Zhao, Antoine Doucet  

**Link**: [PDF](https://arxiv.org/pdf/2507.03350)  

**Abstract**: Sentiment analysis, widely used in product reviews, also impacts financial markets by influencing asset prices through microblogs and news articles. Despite research in sentiment-driven finance, many studies focus on sentence-level classification, overlooking its practical application in trading. This study bridges that gap by evaluating sentiment-based trading strategies for generating positive alpha. We conduct a backtesting analysis using sentiment predictions from three models (two classification and one regression) applied to news articles on Dow Jones 30 stocks, comparing them to the benchmark Buy&Hold strategy. Results show all models produced positive returns, with the regression model achieving the highest return of 50.63% over 28 months, outperforming the benchmark Buy&Hold strategy. This highlights the potential of sentiment in enhancing investment strategies and financial decision-making. 

**Abstract (ZH)**: 基于 sentiment 分析的情感分析广泛应用于产品评价，同时也通过微博和新闻文章影响资本市场，从而影响资产价格。尽管在情感驱动的金融研究中已经有了很多成果，但许多研究着重于句子级别的分类，忽视了其在交易中的实际应用。本研究通过评估基于情感的情感交易策略，填补了这一空白，旨在生成正向阿尔法值。我们使用三种模型（两个分类模型和一个回归模型）对道琼斯 30 种股票的新闻文章进行情感预测，并将其与基准持股策略（Buy&Hold）进行对比。结果显示，所有模型均实现了正收益，其中回归模型在这 28 个月内取得了最高回报率 50.63%，超过了基准持股策略。这表明情感在改进投资策略和金融决策方面具有潜在价值。 

---
# DESign: Dynamic Context-Aware Convolution and Efficient Subnet Regularization for Continuous Sign Language Recognition 

**Title (ZH)**: DESIGN: 动态上下文 Awareness 卷积和高效子网正则化用于连续手语识别 

**Authors**: Sheng Liu, Yiheng Yu, Yuan Feng, Min Xu, Zhelun Jin, Yining Jiang, Tiantian Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2507.03339)  

**Abstract**: Current continuous sign language recognition (CSLR) methods struggle with handling diverse samples. Although dynamic convolutions are ideal for this task, they mainly focus on spatial modeling and fail to capture the temporal dynamics and contextual dependencies. To address this, we propose DESign, a novel framework that incorporates Dynamic Context-Aware Convolution (DCAC) and Subnet Regularization Connectionist Temporal Classification (SR-CTC). DCAC dynamically captures the inter-frame motion cues that constitute signs and uniquely adapts convolutional weights in a fine-grained manner based on contextual information, enabling the model to better generalize across diverse signing behaviors and boost recognition accuracy. Furthermore, we observe that existing methods still rely on only a limited number of frames for parameter updates during training, indicating that CTC learning overfits to a dominant path. To address this, SR-CTC regularizes training by applying supervision to subnetworks, encouraging the model to explore diverse CTC alignment paths and effectively preventing overfitting. A classifier-sharing strategy in SR-CTC further strengthens multi-scale consistency. Notably, SR-CTC introduces no inference overhead and can be seamlessly integrated into existing CSLR models to boost performance. Extensive ablations and visualizations further validate the effectiveness of the proposed methods. Results on mainstream CSLR datasets (i.e., PHOENIX14, PHOENIX14-T, CSL-Daily) demonstrate that DESign achieves state-of-the-art performance. 

**Abstract (ZH)**: 基于动态上下文感知卷积和子网络正则化CTC的连续手语识别框架 

---
# De-Fake: Style based Anomaly Deepfake Detection 

**Title (ZH)**: De-假：基于风格的深度伪造异常检测 

**Authors**: Sudev Kumar Padhi, Harshit Kumar, Umesh Kashyap, Sk. Subidh Ali  

**Link**: [PDF](https://arxiv.org/pdf/2507.03334)  

**Abstract**: Detecting deepfakes involving face-swaps presents a significant challenge, particularly in real-world scenarios where anyone can perform face-swapping with freely available tools and apps without any technical knowledge. Existing deepfake detection methods rely on facial landmarks or inconsistencies in pixel-level features and often struggle with face-swap deepfakes, where the source face is seamlessly blended into the target image or video. The prevalence of face-swap is evident in everyday life, where it is used to spread false information, damage reputations, manipulate political opinions, create non-consensual intimate deepfakes (NCID), and exploit children by enabling the creation of child sexual abuse material (CSAM). Even prominent public figures are not immune to its impact, with numerous deepfakes of them circulating widely across social media platforms. Another challenge faced by deepfake detection methods is the creation of datasets that encompass a wide range of variations, as training models require substantial amounts of data. This raises privacy concerns, particularly regarding the processing and storage of personal facial data, which could lead to unauthorized access or misuse. Our key idea is to identify these style discrepancies to detect face-swapped images effectively without accessing the real facial image. We perform comprehensive evaluations using multiple datasets and face-swapping methods, which showcases the effectiveness of SafeVision in detecting face-swap deepfakes across diverse scenarios. SafeVision offers a reliable and scalable solution for detecting face-swaps in a privacy preserving manner, making it particularly effective in challenging real-world applications. To the best of our knowledge, SafeVision is the first deepfake detection using style features while providing inherent privacy protection. 

**Abstract (ZH)**: 检测涉及面部互换的深fake presents a significant challenge, particularly in real-world scenarios where anyone can perform face-swapping with freely available tools and apps without any technical knowledge。 

---
# Task-Specific Generative Dataset Distillation with Difficulty-Guided Sampling 

**Title (ZH)**: 基于难度引导采样的任务特定生成性数据集蒸馏 

**Authors**: Mingzhuo Li, Guang Li, Jiafeng Mao, Linfeng Ye, Takahiro Ogawa, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2507.03331)  

**Abstract**: To alleviate the reliance of deep neural networks on large-scale datasets, dataset distillation aims to generate compact, high-quality synthetic datasets that can achieve comparable performance to the original dataset. The integration of generative models has significantly advanced this field. However, existing approaches primarily focus on aligning the distilled dataset with the original one, often overlooking task-specific information that can be critical for optimal downstream performance. In this paper, focusing on the downstream task of classification, we propose a task-specific sampling strategy for generative dataset distillation that incorporates the concept of difficulty to consider the requirements of the target task better. The final dataset is sampled from a larger image pool with a sampling distribution obtained by matching the difficulty distribution of the original dataset. A logarithmic transformation is applied as a pre-processing step to correct for distributional bias. The results of extensive experiments demonstrate the effectiveness of our method and suggest its potential for enhancing performance on other downstream tasks. 

**Abstract (ZH)**: 为了减轻深度神经网络对大规模数据集的依赖，数据集蒸馏旨在生成紧凑的、高质量的合成数据集，以便在性能上与原始数据集相当。结合生成模型极大地推进了这一领域的发展。然而，现有方法主要集中在使蒸馏数据集与原始数据集对齐，往往忽略了对最优下游性能至关重要的任务特定信息。在本文中，针对分类下游任务，我们提出了一种嵌入难度概念的任务特定采样策略，以更好地满足目标任务的需求。最终的数据集是从一个更大的图像池中采样得到的，采样分布是通过匹配原始数据集的难度分布获得的。作为预处理步骤，应用对数变换以纠正分布偏差。大量实验的结果表明了该方法的有效性，并暗示其在增强其他下游任务性能方面的潜力。 

---
# Read Quietly, Think Aloud: Decoupling Comprehension and Reasoning in LLMs 

**Title (ZH)**: 静读 aloud, 明理 quietly: 解耦 LLMs 的理解与推理 

**Authors**: Yuanxin Wang, Ganesh Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2507.03327)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable proficiency in understanding text and generating high-quality responses. However, a critical distinction from human cognition is their typical lack of a distinct internal `reading' or deliberation phase before `speaking' (i.e., generating text). Humans often engage in silent reading to comprehend context and formulate thoughts prior to articulation. This paper investigates methods to imbue LLMs with a similar capacity for internal processing.
We introduce and evaluate techniques that encourage LLMs to `read silently.' Our findings indicate that even a straightforward approach, such as providing the model with an initial contextual prompt or `reading space' before it begins predicting subsequent tokens for the final output, can yield significant performance improvements. We further enhance this concept by developing a `reading buddy' architecture, where an auxiliary component silently processes the input and provides refined contextual insights to the primary generation model. These approaches aim to foster deeper understanding from LLMs so that they can produce better reasoned responses, moving them one step closer to more human-like text processing. Our results indicate that these simple techniques can provide surprisingly strong impact on accuracy with multiple point accuracy boost. 

**Abstract (ZH)**: 大型语言模型（LLMs）在理解文本和生成高质量响应方面展现了显著的能力。然而，与人类认知的一个关键区别在于，它们通常在生成文本（即“说话”）之前缺乏明显的内部“阅读”或思考阶段。人类往往会进行无声阅读以理解上下文并形成想法后再表达。本文研究了赋予LLMs类似内部处理能力的方法。我们介绍了并评估了鼓励LLMs进行“无声阅读”的技术。研究结果显示，即使是一种简单的做法，比如在模型开始预测最终输出的后续标记之前提供初始上下文提示或“阅读空间”，也能显著提高性能。我们进一步通过开发一种“阅读伙伴”架构来增强这一概念，其中辅助组件无声地处理输入并为主要内容生成模型提供精细化的上下文洞察。这些方法旨在促进LLMs进行更深层次的理解，从而产生更具道理的响应，使其更接近于类似人类的文本处理。我们的结果表明，这些简单的方法可以显著提高准确性，提供多个点的准确度提升。 

---
# Source-Free Domain Adaptation via Multi-view Contrastive Learning 

**Title (ZH)**: 无源领域适应：多视图对比学习 

**Authors**: Amirfarhad Farhadi, Naser Mozayani, Azadeh Zamanifar  

**Link**: [PDF](https://arxiv.org/pdf/2507.03321)  

**Abstract**: Domain adaptation has become a widely adopted approach in machine learning due to the high costs associated with labeling data. It is typically applied when access to a labeled source domain is available. However, in real-world scenarios, privacy concerns often restrict access to sensitive information, such as fingerprints, bank account details, and facial images. A promising solution to this issue is Source-Free Unsupervised Domain Adaptation (SFUDA), which enables domain adaptation without requiring access to labeled target domain data. Recent research demonstrates that SFUDA can effectively address domain discrepancies; however, two key challenges remain: (1) the low quality of prototype samples, and (2) the incorrect assignment of pseudo-labels. To tackle these challenges, we propose a method consisting of three main phases. In the first phase, we introduce a Reliable Sample Memory (RSM) module to improve the quality of prototypes by selecting more representative samples. In the second phase, we employ a Multi-View Contrastive Learning (MVCL) approach to enhance pseudo-label quality by leveraging multiple data augmentations. In the final phase, we apply a noisy label filtering technique to further refine the pseudo-labels. Our experiments on three benchmark datasets - VisDA 2017, Office-Home, and Office-31 - demonstrate that our method achieves approximately 2 percent and 6 percent improvements in classification accuracy over the second-best method and the average of 13 well-known state-of-the-art approaches, respectively. 

**Abstract (ZH)**: 源域无标签的领域自适应方法：解决原型样本质量低和伪标签误配问题 

---
# Structure-Aware Compound-Protein Affinity Prediction via Graph Neural Network with Group Lasso Regularization 

**Title (ZH)**: 基于图神经网络和分组lasso正则化的结构感知化合物-蛋白亲和力预测 

**Authors**: Zanyu Shi, Yang Wang, Pathum Weerawarna, Jie Zhang, Timothy Richardson, Yijie Wang, Kun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03318)  

**Abstract**: Explainable artificial intelligence (XAI) approaches have been increasingly applied in drug discovery to learn molecular representations and identify substructures driving property predictions. However, building end-to-end explainable machine learning models for structure-activity relationship (SAR) modeling for compound property prediction faces many challenges, such as limited activity data per target and the sensitivity of properties to subtle molecular changes. To address this, we leveraged activity-cliff molecule pairs, i.e., compounds sharing a common scaffold but differing sharply in potency, targeting three proto-oncogene tyrosine-protein kinase Src proteins (i.e., PDB IDs 1O42, 2H8H, and 4MXO). We implemented graph neural network (GNN) methods to obtain atom-level feature information and predict compound-protein affinity (i.e., half maximal inhibitory concentration, IC50). In addition, we trained GNN models with different structure-aware loss functions to adequately leverage molecular property and structure information. We also utilized group lasso and sparse group lasso to prune and highlight molecular subgraphs and enhance the structure-specific model explainability for the predicted property difference in molecular activity-cliff pairs. We improved drug property prediction by integrating common and uncommon node information and using sparse group lasso, reducing the average root mean squared error (RMSE) by 12.70%, and achieving the lowest averaged RMSE=0.2551 and the highest PCC=0.9572. Furthermore, applying regularization enhances feature attribution methods that estimate the contribution of each atom in the molecular graphs by boosting global direction scores and atom-level accuracy in atom coloring accuracy, which improves model interpretability in drug discovery pipelines, particularly in investigating important molecular substructures in lead optimization. 

**Abstract (ZH)**: 可解释的人工智能（XAI）方法在药物发现中的应用：通过活性悬崖分子对构建化合物活性预测的结构-活性关系（SAR）模型 

---
# Partial Label Learning for Automated Theorem Proving 

**Title (ZH)**: 部分标签学习在自动定理证明中的应用 

**Authors**: Zsolt Zombori, Balázs Indruck  

**Link**: [PDF](https://arxiv.org/pdf/2507.03314)  

**Abstract**: We formulate learning guided Automated Theorem Proving as Partial Label Learning, building the first bridge across these fields of research and providing a theoretical framework for dealing with alternative proofs during learning. We use the plCoP theorem prover to demonstrate that methods from the Partial Label Learning literature tend to increase the performance of learning assisted theorem provers. 

**Abstract (ZH)**: 我们将学习引导的自动定理证明形式化为部分标签学习，建立了这些研究领域的第一座桥梁，并提供了在学习过程中处理替代证明的理论框架。我们使用plCoP定理证明器证明，部分标签学习文献中的方法倾向于提高学习辅助定理证明器的性能。 

---
# Personalized Image Generation from an Author Writing Style 

**Title (ZH)**: 根据作者写作风格的个性化图像生成 

**Authors**: Sagar Gandhi, Vishal Gandhi  

**Link**: [PDF](https://arxiv.org/pdf/2507.03313)  

**Abstract**: Translating nuanced, textually-defined authorial writing styles into compelling visual representations presents a novel challenge in generative AI. This paper introduces a pipeline that leverages Author Writing Sheets (AWS) - structured summaries of an author's literary characteristics - as input to a Large Language Model (LLM, Claude 3.7 Sonnet). The LLM interprets the AWS to generate three distinct, descriptive text-to-image prompts, which are then rendered by a diffusion model (Stable Diffusion 3.5 Medium). We evaluated our approach using 49 author styles from Reddit data, with human evaluators assessing the stylistic match and visual distinctiveness of the generated images. Results indicate a good perceived alignment between the generated visuals and the textual authorial profiles (mean style match: $4.08/5$), with images rated as moderately distinctive. Qualitative analysis further highlighted the pipeline's ability to capture mood and atmosphere, while also identifying challenges in representing highly abstract narrative elements. This work contributes a novel end-to-end methodology for visual authorial style personalization and provides an initial empirical validation, opening avenues for applications in creative assistance and cross-modal understanding. 

**Abstract (ZH)**: 将文本定义的作者写作风格微妙的表达转化为引人入胜的视觉表现构成了生成式AI的新挑战。本论文引入了一种基于作者写作表（AWS）的流程，AWS是对作者文学特征的结构化总结，作为大型语言模型（LLM，Claude 3.7 Sonnet）的输入。LLM通过解释AWS生成三个不同的、描述性的文本到图像提示，这些提示随后由扩散模型（Stable Diffusion 3.5 Medium）呈现。我们使用来自Reddit的数据中的49种作者风格进行了评估，由人类评估者评估生成图像的风格匹配度和视觉独特性。结果表明，生成的视觉效果与文本作者特征（平均风格匹配：4.08/5）之间存在良好的感知契合度，并且图像被评价为中等独特性。定性分析进一步强调了该流程捕捉氛围和情绪的能力，同时也指出了在表现高度抽象的叙事元素方面的挑战。本研究贡献了一种新颖的整体方法，用于视觉作者风格个性化，并提供了初步的经验验证，开启了在创造性辅助和跨模态理解方面的应用途径。 

---
# GRAFT: A Graph-based Flow-aware Agentic Framework for Document-level Machine Translation 

**Title (ZH)**: 基于图的流敏代理框架：面向文档级机器翻译 

**Authors**: Himanshu Dutta, Sunny Manchanda, Prakhar Bapat, Meva Ram Gurjar, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2507.03311)  

**Abstract**: Document level Machine Translation (DocMT) approaches often struggle with effectively capturing discourse level phenomena. Existing approaches rely on heuristic rules to segment documents into discourse units, which rarely align with the true discourse structure required for accurate translation. Otherwise, they fail to maintain consistency throughout the document during translation. To address these challenges, we propose Graph Augmented Agentic Framework for Document Level Translation (GRAFT), a novel graph based DocMT system that leverages Large Language Model (LLM) agents for document translation. Our approach integrates segmentation, directed acyclic graph (DAG) based dependency modelling, and discourse aware translation into a cohesive framework. Experiments conducted across eight translation directions and six diverse domains demonstrate that GRAFT achieves significant performance gains over state of the art DocMT systems. Specifically, GRAFT delivers an average improvement of 2.8 d BLEU on the TED test sets from IWSLT2017 over strong baselines and 2.3 d BLEU for domain specific translation from English to Chinese. Moreover, our analyses highlight the consistent ability of GRAFT to address discourse level phenomena, yielding coherent and contextually accurate translations. 

**Abstract (ZH)**: 基于图增强自主框架的文档级别机器翻译（GRAFT） 

---
# ReTimeCausal: EM-Augmented Additive Noise Models for Interpretable Causal Discovery in Irregular Time Series 

**Title (ZH)**: ReTimeCausal：用于不规则时间序列可解释因果发现的EM增强加性噪声模型 

**Authors**: Weihong Li, Anpeng Wu, Kun Kuang, Keting Yin  

**Link**: [PDF](https://arxiv.org/pdf/2507.03310)  

**Abstract**: This paper studies causal discovery in irregularly sampled time series-a pivotal challenge in high-stakes domains like finance, healthcare, and climate science, where missing data and inconsistent sampling frequencies distort causal mechanisms. Traditional methods (e.g., Granger causality, PCMCI) fail to reconcile multi-scale interactions (e.g., hourly storms vs. decadal climate shifts), while neural approaches (e.g., CUTS+) lack interpretability, stemming from a critical gap: existing frameworks either rigidly assume temporal regularity or aggregate dynamics into opaque representations, neglecting real-world granularity and auditable logic. To bridge this gap, we propose ReTimeCausal, a novel integration of Additive Noise Models (ANM) and Expectation-Maximization (EM) that unifies physics-guided data imputation with sparse causal inference. Through kernelized sparse regression and structural constraints, ReTimeCausal iteratively refines missing values (E-step) and causal graphs (M-step), resolving cross-frequency dependencies and missing data issues. Extensive experiments on synthetic and real-world datasets demonstrate that ReTimeCausal outperforms existing state-of-the-art methods under challenging irregular sampling and missing data conditions. 

**Abstract (ZH)**: 这项研究探讨了不规则采样时间序列的因果发现——在金融、医疗和气候科学等高风险领域中一个关键挑战，其中缺失数据和不一致的采样频率扭曲了因果机制。传统方法（例如Granger因果关系、PCMCI）无法解决多尺度交互（例如小时级风暴与十年级气候变化）的问题，而神经方法（例如CUTS+）缺乏可解释性，源自于现有框架的关键缺陷：它们要么严格假设时间的规律性，要么将动态聚合为不透明的表现形式，忽视了现实世界的细节和可审计的逻辑。为了解决这一缺陷，我们提出了一种名为ReTimeCausal的新颖集成方法，它结合了加性噪声模型（ANM）和期望最大化（EM），统一了基于物理的数据插补与稀疏因果推断。通过核稀疏回归和结构约束，ReTimeCausal迭代地完善缺失值（E步）和因果图（M步），解决跨频率依赖性和缺失数据问题。在合成数据集和真实世界数据集上的广泛实验证明，在面对不规则采样和缺失数据的挑战条件下，ReTimeCausal优于现有最先进的方法。 

---
# Scaffolding Recursive Divergence and Convergence in Story Ideation 

**Title (ZH)**: 支撑递归发散与收敛在故事构思中的应用 

**Authors**: Taewook Kim, Matthew Kay, Yuqian Sun, Melissa Roemmele, Max Kreminski, John Joon Young Chung  

**Link**: [PDF](https://arxiv.org/pdf/2507.03307)  

**Abstract**: Human creative ideation involves both exploration of diverse ideas (divergence) and selective synthesis of explored ideas into coherent combinations (convergence). While processes of divergence and convergence are often interleaved and nested, existing AI-powered creativity support tools (CSTs) lack support for sophisticated orchestration of divergence and convergence. We present Reverger, an AI-powered CST that helps users ideate variations of conceptual directions for modifying a story by scaffolding flexible iteration between divergence and convergence. For divergence, our tool enables recursive exploration of alternative high-level directions for modifying a specific part of the original story. For convergence, it allows users to collect explored high-level directions and synthesize them into concrete variations. Users can then iterate between divergence and convergence until they find a satisfactory outcome. A within-subject study revealed that Reverger permitted participants to explore more unexpected and diverse high-level directions than a comparable baseline. Reverger users also felt that they had more fine-grained control and discovered more effort-worthy outcomes. 

**Abstract (ZH)**: 人类创造性构思涉及对多样化想法的探索（发散）和对探索过的想法进行选择性综合以形成连贯组合（收敛）。虽然发散和收敛的过程通常是交错和嵌套的，但现有的基于AI的创意支持工具（CSTs）缺乏对发散和收敛复杂 orchestration 的支持。我们介绍了一种基于AI的CST——Reverger，它通过在发散和收敛之间的灵活迭代来帮助用户构思修改故事的概念方向。在发散阶段，我们的工具支持对特定部分原始故事进行替代的高层次方向的递归探索。在收敛阶段，它允许用户收集探索过的高层次方向并将其综合成具体的变化。用户可以在此发散和收敛之间迭代，直到找到令人满意的结果。一项以同一受试者为对象的研究显示，Reverger 允许参与者比一个可比的 baseline 涉及更多意想不到和多样的高层次方向。Reverger 的用户还感觉到他们有更多的细腻控制，并发现了更多值得付出努力的结果。 

---
# Leveraging Out-of-Distribution Unlabeled Images: Semi-Supervised Semantic Segmentation with an Open-Vocabulary Model 

**Title (ZH)**: 利用分布外未标注图像：基于开放词汇表模型的半监督语义分割 

**Authors**: Wooseok Shin, Jisu Kang, Hyeonki Jeong, Jin Sob Kim, Sung Won Han  

**Link**: [PDF](https://arxiv.org/pdf/2507.03302)  

**Abstract**: In semi-supervised semantic segmentation, existing studies have shown promising results in academic settings with controlled splits of benchmark datasets. However, the potential benefits of leveraging significantly larger sets of unlabeled images remain unexplored. In real-world scenarios, abundant unlabeled images are often available from online sources (web-scraped images) or large-scale datasets. However, these images may have different distributions from those of the target dataset, a situation known as out-of-distribution (OOD). Using these images as unlabeled data in semi-supervised learning can lead to inaccurate pseudo-labels, potentially misguiding network training. In this paper, we propose a new semi-supervised semantic segmentation framework with an open-vocabulary segmentation model (SemiOVS) to effectively utilize unlabeled OOD images. Extensive experiments on Pascal VOC and Context datasets demonstrate two key findings: (1) using additional unlabeled images improves the performance of semi-supervised learners in scenarios with few labels, and (2) using the open-vocabulary segmentation (OVS) model to pseudo-label OOD images leads to substantial performance gains. In particular, SemiOVS outperforms existing PrevMatch and SemiVL methods by +3.5 and +3.0 mIoU, respectively, on Pascal VOC with a 92-label setting, achieving state-of-the-art performance. These findings demonstrate that our approach effectively utilizes abundant unlabeled OOD images for semantic segmentation tasks. We hope this work can inspire future research and real-world applications. The code is available at this https URL 

**Abstract (ZH)**: 半监督语义分割中，现有研究在受控划分的标准数据集中展示了令人鼓舞的结果。然而，利用显著更大的未标注图像集的潜在优势尚未被探索。在实际场景中，从网上抓取的图像或大型数据集中往往可以获得大量的未标注图像。然而，这些图像的分布可能与目标数据集不同，这种情况被称为离分布（OOD）。将这些图像作为未标注数据用于半监督学习可能会导致不准确的伪标签，从而可能误导网络训练。本文提出了一种新的半监督语义分割框架（SemiOVS），该框架配备了一个开放词汇量分割模型，以有效利用离分布的未标注图像。在Pascal VOC和Context数据集上的广泛实验表明，两个关键发现：（1）使用额外的未标注图像可以提高少量标签场景下半监督学习器的性能；（2）使用开放词汇量分割（OVS）模型为离分布图像伪标签可以带来显著的性能提升。特别是，SemiOVS在Pascal VOC 92标签设置下的mIoU上分别比PrevMatch和SemiVL方法高出3.5和3.0，达到了最先进的性能。这些发现表明，该方法有效地利用了大量的离分布未标注图像来完成语义分割任务。我们希望这项工作能激发未来的研究和实际应用。代码可在以下链接获取。 

---
# MGAA: Multi-Granular Adaptive Allocation fof Low-Rank Compression of LLMs 

**Title (ZH)**: MGAA: 多粒度自适应分配用于大语言模型低秩压缩 

**Authors**: Guangyan Li, Yongqiang Tang, Wensheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03294)  

**Abstract**: The enormous parameter scale of large language models (LLMs) has made model compression a research hotspot, which aims to alleviate computational resource demands during deployment and inference. As a promising direction, low-rank approximation technique has made remarkable achievements. Nevertheless, unfortunately, the vast majority of studies to low-rank approximation compression generally apply uniform compression ratios across all weight matrices, while disregarding their inherently differentiated impacts on the model's performance. Although a few recent work attempts to employ heuristic search strategies to achieve the optimal parameter allocation, such strategies are computationally inefficient and lose the generalization ability in the era of LLMs. In this study, we propose a novel parameter Multi-Granular Adaptive Allocation (MGAA) method, which can adaptively allocate parameters between and within sublayers without task-specific evaluations in the compression process. MGAA consists of two components: 1) Among different sublayers, it assigns compression ratios based on their cosine similarity between inputs and outputs, allowing for a more tailored compression in sublayers with varying degrees of importance, and 2) Within each sublayer, it allocates different compression ratios to weight matrices based on their energy distribution characteristics, ensuring a consistent energy retention ratio while optimizing compression efficiency. Comprehensive evaluations of MGAA across multiple LLMs backbone models and benchmark datasets demonstrate its superior performance. Additionally, we apply our MGAA to multimodal model LLaVA, exhibiting remarkable performance improvements. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的巨大参数规模使得模型压缩成为研究热点，旨在部署和推理过程中缓解计算资源需求。作为一种有前途的方向，低秩逼近技术已经取得了显著成就。然而，大多数关于低秩逼近压缩的研究通常在所有权重矩阵上应用统一的压缩比率，而忽视了它们对模型性能的不同影响。虽然一些最近的工作尝试使用启发式搜索策略来实现最优参数分配，但这些策略在大规模语言模型时代计算效率低下，并且丧失了泛化能力。在本研究中，我们提出了一种新型参数多粒度自适应分配（MGAA）方法，在压缩过程中无需针对特定任务进行参数分配。MGAA由两个部分组成：1）在不同的子层之间，根据输入和输出之间的余弦相似度分配压缩比率，从而在不同重要程度的子层中实现更加个性化的压缩；2）在每个子层内部，基于权重矩阵的能量分布特性分配不同的压缩比率，确保能量保留比的一致性同时优化压缩效率。MGAA在多个LLM骨干模型和基准数据集上的综合评估展示了其优越性能。此外，我们将MGAA应用于多模态模型LLaVA，显现出显著的性能提升。 

---
# Conformal Information Pursuit for Interactively Guiding Large Language Models 

**Title (ZH)**: 符合信息引导的大语言模型交互式辅助方法 

**Authors**: Kwan Ho Ryan Chan, Yuyan Ge, Edgar Dobriban, Hamed Hassani, René Vidal  

**Link**: [PDF](https://arxiv.org/pdf/2507.03279)  

**Abstract**: A significant use case of instruction-finetuned Large Language Models (LLMs) is to solve question-answering tasks interactively. In this setting, an LLM agent is tasked with making a prediction by sequentially querying relevant information from the user, as opposed to a single-turn conversation. This paper explores sequential querying strategies that aim to minimize the expected number of queries. One such strategy is Information Pursuit (IP), a greedy algorithm that at each iteration selects the query that maximizes information gain or equivalently minimizes uncertainty. However, obtaining accurate estimates of mutual information or conditional entropy for LLMs is very difficult in practice due to over- or under-confident LLM probabilities, which leads to suboptimal query selection and predictive performance. To better estimate the uncertainty at each iteration, we propose Conformal Information Pursuit (C-IP), an alternative approach to sequential information gain based on conformal prediction sets. More specifically, C-IP leverages a relationship between prediction sets and conditional entropy at each iteration to estimate uncertainty based on the average size of conformal prediction sets. In contrast to conditional entropy, we find that conformal prediction sets are a distribution-free and robust method of measuring uncertainty. Experiments with 20 Questions show that C-IP obtains better predictive performance and shorter query-answer chains compared to previous approaches to IP and uncertainty-based chain-of-thought methods. Furthermore, extending to an interactive medical setting between a doctor and a patient on the MediQ dataset, C-IP achieves competitive performance with direct single-turn prediction while offering greater interpretability. 

**Abstract (ZH)**: 大型语言模型（LLMs）指令调优的重要应用场景是解决交互式问答任务。在这种设置中，LLM代理需要通过顺序查询相关信息来自动生成预测，而不是进行单轮对话。本文探索旨在最小化预期查询次数的顺序查询策略。其中一种策略是信息追求（IP），这是一种贪婪算法，在每一步迭代中选择最大化信息增益或等价地最小化不确定性的问题查询。然而，由于LLM概率的高估或低估，实际中很难获得准确的互信息或条件熵估计，这会导致次优的查询选择和预测性能。为了更好地在每一步迭代中估计不确定性，我们提出了聚合法信息追求（C-IP），这是一种基于聚合法预测集的顺序信息增益的替代方法。具体而言，C-IP 利用每一步迭代中预测集和条件熵之间的关系，通过聚合法预测集的平均大小来估计不确定性。与条件熵不同，我们发现聚合法预测集是一种无分布且稳健的不确定性度量方法。20 问题实验表明，C-IP 在预测性能和较短的查询-回答链方面优于之前的 IP 方法和基于不确定性的链式思考方法。此外，将 C-IP 拓展到医生和患者之间的互动医疗场景（使用 MediQ 数据集），C-IP 在提供更强的可解释性的同时，实现了与直接单轮预测相当的竞争力。 

---
# Investigating Redundancy in Multimodal Large Language Models with Multiple Vision Encoders 

**Title (ZH)**: 探讨多模态大型语言模型中多个视觉编码器的冗余性 

**Authors**: Song Mao, Yang Chen, Pinglong Cai, Ding Wang, Guohang Yan, Zhi Yu, Botian Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.03262)  

**Abstract**: Multimodal Large Language Models (MLLMs) increasingly adopt multiple vision encoders to capture diverse visual information, ranging from coarse semantics to fine grained details. While this approach is intended to enhance visual understanding capability, we observe that the performance gains from adding encoders often diminish and can even lead to performance degradation, a phenomenon we term encoder redundancy. This paper presents a systematic investigation into this issue. Through comprehensive ablation studies on state of the art multi encoder MLLMs, we empirically demonstrate that significant redundancy exists. To quantify each encoder's unique contribution, we propose a principled metric: the Conditional Utilization Rate (CUR). Building on CUR, we introduce the Information Gap (IG) to capture the overall disparity in encoder utility within a this http URL experiments reveal that certain vision encoders contribute little, or even negatively, to overall performance, confirming substantial redundancy. Our experiments reveal that certain vision encoders contribute minimally, or even negatively, to the model's performance, confirming the prevalence of redundancy. These findings highlight critical inefficiencies in current multi encoder designs and establish that our proposed metrics can serve as valuable diagnostic tools for developing more efficient and effective multimodal architectures. 

**Abstract (ZH)**: 多模态大型语言模型中的编码器冗余现象及其量化研究 

---
# ForgeHLS: A Large-Scale, Open-Source Dataset for High-Level Synthesis 

**Title (ZH)**: ForgeHLS：一种大规模开源高阶综合数据集 

**Authors**: Zedong Peng, Zeju Li, Mingzhe Gao, Qiang Xu, Chen Zhang, Jieru Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.03255)  

**Abstract**: We introduce ForgeEDA, an open-source comprehensive circuit dataset across various categories. ForgeEDA includes diverse circuit representations such as Register Transfer Level (RTL) code, Post-mapping (PM) netlists, And-Inverter Graphs (AIGs), and placed netlists, enabling comprehensive analysis and development. We demonstrate ForgeEDA's utility by benchmarking state-of-the-art EDA algorithms on critical tasks such as Power, Performance, and Area (PPA) optimization, highlighting its ability to expose performance gaps and drive advancements. Additionally, ForgeEDA's scale and diversity facilitate the training of AI models for EDA tasks, demonstrating its potential to improve model performance and generalization. By addressing limitations in existing datasets, ForgeEDA aims to catalyze breakthroughs in modern IC design and support the next generation of innovations in EDA. 

**Abstract (ZH)**: 我们介绍ForgeEDA，一个涵盖多种类别开源综合电路数据集。ForgeEDA包括多种电路表示形式，如寄存器传输级（RTL）代码、映射后（PM）网表、与门-反相器图（AIGs）和布线后网表，支持全面分析和开发。通过在关键任务如功率、性能和面积（PPA）优化上 benchmark 现有的EDA算法，展示了其揭示性能差距并推动进步的能力。此外，ForgeEDA的规模和多样性使其适用于EDA任务中的AI模型训练，展示了其提高模型性能和泛化的潜力。通过弥补现有数据集的不足，ForgeEDA旨在推动现代IC设计的突破，并支持下一代EDA创新。 

---
# RefineX: Learning to Refine Pre-training Data at Scale from Expert-Guided Programs 

**Title (ZH)**: RefineX: 从专家指导程序中大规模精炼预训练数据 

**Authors**: Baolong Bi, Shenghua Liu, Xingzhang Ren, Dayiheng Liu, Junyang Lin, Yiwei Wang, Lingrui Mei, Junfeng Fang, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.03253)  

**Abstract**: The foundational capabilities of large language models (LLMs) are deeply influenced by the quality of their pre-training corpora. However, enhancing data quality at scale remains a significant challenge, primarily due to the trade-off between refinement effectiveness and processing efficiency. While rule-based filtering remains the dominant paradigm, it typically operates at the document level and lacks the granularity needed to refine specific content within documents. Inspired by emerging work such as ProX, we propose $\textbf{RefineX}$, a novel framework for large-scale, surgical refinement of pre-training data through programmatic editing tasks. RefineX enables efficient and fine-grained data refinement while reliably preserving the diversity and naturalness of raw text. The core strength of RefineX lies in distilling high-quality, expert-guided end-to-end refinement results into minimal edit-based deletion programs. This high-precision distillation pipeline is used to train an efficient and reliable refine model that can systematically improve every instance in the corpus at scale. We evaluate RefineX across from-scratch pre-training at multiple model scales and find that it consistently outperforms models trained on raw, filtered, or alternatively refined data across diverse downstream tasks. On the 750M model, RefineX yields 2.6%-7.2% average gains on lighteval tasks, and achieves comparable performance using significantly fewer training tokens. Further analysis shows that RefineX reliably enhances text quality with both high efficiency and precision, outperforming prior approaches such as end-to-end generation and Prox-C. These results position RefineX as a scalable, effective, and reliable solution for optimizing pre-training data in modern LLM pipelines. 

**Abstract (ZH)**: 大规模语言模型的基礎能力受其前期训练语料质量的影响。然而，大规模提升数据质量仍然是一项重大挑战，主要由于精炼效果与处理效率之间的权衡。尽管基于规则的过滤仍然是主导范式，但通常在文档级别操作，缺乏对文档内具体内容进行精细化精炼所需的粒度。受ProX等新兴工作的启发，我们提出了**RefineX**框架，这是一种用于大规模、手术式精炼预训练数据的新颖程序化编辑任务框架。RefineX使数据精炼既高效又精细化，同时可靠地保留原始文本的多样性和自然性。RefineX的核心优势在于将高质量、专家导向的一体化精炼结果精炼为最小的基于编辑的删除程序。这一高精度精炼管道用于训练高效的可靠精炼模型，可以大规模系统地改进语料库中的每一个实例。我们在多个模型规模的从头预训练中评估了RefineX，发现它在多样化的下游任务中始终优于使用未过滤、过滤或替代精炼数据训练的模型。在750M模型上，RefineX在轻量评估任务中平均提高了2.6%至7.2%，并且使用显著较少的训练标记达到类似性能。进一步分析表明，RefineX在高效率和高精度下可靠提升文本质量，优于先前方法如端到端生成和Prox-C。这些结果将RefineX定位为一种在现代大规模语言模型管道中优化预训练数据的大规模、有效且可靠解决方案。 

---
# Toward Efficient Speech Emotion Recognition via Spectral Learning and Attention 

**Title (ZH)**: 基于频谱学习和注意力机制的高效语音情感识别 

**Authors**: HyeYoung Lee, Muhammad Nadeem  

**Link**: [PDF](https://arxiv.org/pdf/2507.03251)  

**Abstract**: Speech Emotion Recognition (SER) traditionally relies on auditory data analysis for emotion classification. Several studies have adopted different methods for SER. However, existing SER methods often struggle to capture subtle emotional variations and generalize across diverse datasets. In this article, we use Mel-Frequency Cepstral Coefficients (MFCCs) as spectral features to bridge the gap between computational emotion processing and human auditory perception. To further improve robustness and feature diversity, we propose a novel 1D-CNN-based SER framework that integrates data augmentation techniques. MFCC features extracted from the augmented data are processed using a 1D Convolutional Neural Network (CNN) architecture enhanced with channel and spatial attention mechanisms. These attention modules allow the model to highlight key emotional patterns, enhancing its ability to capture subtle variations in speech signals. The proposed method delivers cutting-edge performance, achieving the accuracy of 97.49% for SAVEE, 99.23% for RAVDESS, 89.31% for CREMA-D, 99.82% for TESS, 99.53% for EMO-DB, and 96.39% for EMOVO. Experimental results show new benchmarks in SER, demonstrating the effectiveness of our approach in recognizing emotional expressions with high precision. Our evaluation demonstrates that the integration of advanced Deep Learning (DL) methods substantially enhances generalization across diverse datasets, underscoring their potential to advance SER for real-world deployment in assistive technologies and human-computer interaction. 

**Abstract (ZH)**: 基于1D-CNN的情感语音识别方法：利用梅尔频谱系数和数据增强技术 

---
# On Jailbreaking Quantized Language Models Through Fault Injection Attacks 

**Title (ZH)**: 通过故障注入攻击破解量化语言模型 

**Authors**: Noureldin Zahran, Ahmad Tahmasivand, Ihsen Alouani, Khaled Khasawneh, Mohammed E. Fouda  

**Link**: [PDF](https://arxiv.org/pdf/2507.03236)  

**Abstract**: The safety alignment of Language Models (LMs) is a critical concern, yet their integrity can be challenged by direct parameter manipulation attacks, such as those potentially induced by fault injection. As LMs are increasingly deployed using low-precision quantization for efficiency, this paper investigates the efficacy of such attacks for jailbreaking aligned LMs across different quantization schemes. We propose gradient-guided attacks, including a tailored progressive bit-level search algorithm introduced herein and a comparative word-level (single weight update) attack. Our evaluation on Llama-3.2-3B, Phi-4-mini, and Llama-3-8B across FP16 (baseline), and weight-only quantization (FP8, INT8, INT4) reveals that quantization significantly influences attack success. While attacks readily achieve high success (>80\% Attack Success Rate, ASR) on FP16 models, within an attack budget of 25 perturbations, FP8 and INT8 models exhibit ASRs below 20\% and 50\%, respectively. Increasing the perturbation budget up to 150 bit-flips, FP8 models maintained ASR below 65\%, demonstrating some resilience compared to INT8 and INT4 models that have high ASR. In addition, analysis of perturbation locations revealed differing architectural targets across quantization schemes, with (FP16, INT4) and (INT8, FP8) showing similar characteristics. Besides, jailbreaks induced in FP16 models were highly transferable to subsequent FP8/INT8 quantization (<5\% ASR difference), though INT4 significantly reduced transferred ASR (avg. 35\% drop). These findings highlight that while common quantization schemes, particularly FP8, increase the difficulty of direct parameter manipulation jailbreaks, vulnerabilities can still persist, especially through post-attack quantization. 

**Abstract (ZH)**: 语言模型的安全对齐安全性是一个关键问题，然而它们的完整性可能受到直接参数操作攻击的挑战，如由故障注入引发的攻击。随着语言模型越来越多地通过低精度量化来提高效率进行部署，本文探讨了不同量化方案下此类攻击突破对齐语言模型的有效性。我们提出了梯度引导攻击，包括一种在此引入的定制分阶段位级搜索算法和单一权重更新的词级比较攻击。在对Llama-3.2-3B、Phi-4-mini和Llama-3-8B进行的评估中，包括在FP16（基线）、权重唯一量化（FP8、INT8、INT4）下显示，量化显著影响攻击成功率。虽然攻击预算为25个扰动时，FP16模型的攻击成功率超过80%，FP8和INT8模型分别低于20%和50%。将扰动预算增加到150位翻转时，FP8模型的攻击成功率保持在65%以下，显示出比INT8和INT4模型更高的鲁棒性。此外，扰动位置的分析显示不同量化方案下有不同的架构目标，(FP16, INT4)和(INT8, FP8)表现出相似特性。此外，在FP16模型中诱导的突破在随后的FP8/INT8量化中高度可转移（<5%的攻击成功率差异），尽管INT4显著降低了转移的攻击成功率（平均下降35%）。这些发现表明，虽然常见的量化方案，特别是FP8，增加了直接参数操作突破的难度，但漏洞仍然可能存在，尤其是在攻击后的量化过程中。 

---
# The role of gain neuromodulation in layer-5 pyramidal neurons 

**Title (ZH)**: 层5/py层5锥形神经元的增益神经调节作用 

**Authors**: Alejandro Rodriguez-Garcia, Christopher J. Whyte, Brandon R. Munn, Jie Mei, James M. Shine, Srikanth Ramaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2507.03222)  

**Abstract**: Biological and artificial learning systems alike confront the plasticity-stability dilemma. In the brain, neuromodulators such as acetylcholine and noradrenaline relieve this tension by tuning neuronal gain and inhibitory gating, balancing segregation and integration of circuits. Fed by dense cholinergic and noradrenergic projections from the ascending arousal system, layer-5 pyramidal neurons in the cerebral cortex offer a relevant substrate for understanding these dynamics. When distal dendritic signals coincide with back-propagating action potentials, calcium plateaus turn a single somatic spike into a high-gain burst, and interneuron inhibition sculpts the output. These properties make layer-5 cells gain-tunable amplifiers that translate neuromodulatory cues into flexible cortical activity. To capture this mechanism we developed a two-compartment Izhikevich model for pyramidal neurons and single-compartment somatostatin (SOM) and parvalbumin (PV) interneurons, linked by Gaussian connectivity and spike-timing-dependent plasticity (STDP). The soma and apical dendrite are so coupled that somatic spikes back-propagate, while dendritic plateaus can switch the soma from regular firing to bursting by shifting reset and adaptation variables. We show that stronger dendritic drive or tighter coupling raise gain by increasing the likelihood of calcium-triggered somatic bursts. In contrast, dendritic-targeted inhibition suppresses gain, while somatic-targeted inhibition raises the firing threshold of neighboring neurons, thus gating neurons output. Notably, bursting accelerates STDP, supporting rapid synaptic reconfiguration and this http URL suggests that brief gain pulses driven by neuromodulators could serve as an adaptive two-timescale optimization mechanism, effectively modulating the synaptic weight updates. 

**Abstract (ZH)**: 生物学和人工学习系统都面临塑性-稳定性 dilemma。在大脑中，乙酰胆碱和去甲肾上腺素等神经调节物通过调节神经元增益和抑制性门控，平衡电路的分离与整合，缓解这一矛盾。由上行觉醒系统丰富的乙酰胆碱能和去甲肾上腺素能投射供养的皮层层-5锥体细胞，提供了理解这些动态的合适模型。当远端树突信号与后传动作电位 coincidence 时，钙平台将单个胞体尖峰转化为高增益爆发，而抑制性中间神经元则塑造输出。这些特性使层-5细胞成为增益可调的放大器，将神经调节线索转化为灵活的皮层活动。为了捕捉这一机制，我们开发了一种双室Izhikevich模型，用于锥体细胞和单室somatostatin (SOM) 和parvalbumin (PV) 中间神经元，并通过高斯连接和长时程塑性（STDP）将它们联系起来。细胞体与树突高度耦合，使后传胞体尖峰成为可能，同时树突平台可通过调整重置和适应变量将胞体从规则放电切换为爆发放电。研究显示，更强的树突驱动或更紧密的耦合通过增加钙触发胞体爆发的可能性来提高增益。相反，树突目标抑制抑制增益，而胞体目标抑制则提高邻近神经元的放电阈值，从而控制神经元输出。值得注意的是，爆发性活动加速了STDP，支持快速的突触重构。这一结果表明，由神经调节物驱动的短暂增益脉冲可能作为一种适应性两时标优化机制，有效调节突触权重更新，从而实现这一机制。 

---
# Neural Inhibition Improves Dynamic Routing and Mixture of Experts 

**Title (ZH)**: 神经抑制优化动态路由和专家混合 

**Authors**: Will Y. Zou, Jennifer Y. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03221)  

**Abstract**: To be effective, efficient, and diverse, deep learning models need to dynamically choose its architecture based on signals from a population of neurons. We hypothesize dynamic routing models can be improved with neural inhibition in those neural populations. This means signals commonly shared among the various modes of data statistics can be inhibited so that the routing model can choose a specialized expert path for each data sample. Only through inhibition is the routing mechanism able to effectively select neural pathways. We believe this is an under-studied and under-verified implementation methodology for Mixture-of-Experts, dynamic routing, and transformer language models. We provide experimental evidence that the neural inhibition algorithm significantly boosts the performance of general tasks and motivates more effort to be invested in this research direction. 

**Abstract (ZH)**: 深度学习模型需要根据神经群体的信号动态选择其架构以实现有效、高效和多样化。我们假设在这些神经群体中加入神经抑制可以改进动态路由模型。这意味着可以抑制多种数据统计模式下共有的信号，从而使路由模型为每个数据样本选择一个专门的专家路径。只有通过抑制，路由机制才能有效地选择神经路径。我们认为这是一项对混合专家模型、动态路由和变换器语言模型研究值得进一步探索和验证的方法。我们提供了实验证据，证明神经抑制算法显著提升了通用任务的性能，并激发了更多对该研究方向的关注和投入。 

---
# Symbiosis: Multi-Adapter Inference and Fine-Tuning 

**Title (ZH)**: 共生：多适配器推理与微调 

**Authors**: Saransh Gupta, Umesh Deshpande, Travis Janssen, Swami Sundararaman  

**Link**: [PDF](https://arxiv.org/pdf/2507.03220)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) allows model builders to capture the task specific parameters into adapters, which are a fraction of the size of the original base model. Popularity of PEFT technique for fine-tuning has led to creation of a large number of adapters for popular Large Language Models (LLMs). However, existing frameworks fall short in supporting inference or fine-tuning with multiple adapters in the following ways. 1) For fine-tuning, each job needs to deploy its dedicated base model instance, which results in excessive GPU memory consumption and poor GPU utilization. 2) While popular inference platforms can serve multiple PEFT adapters, they do not allow independent resource management or mixing of different PEFT methods. 3) They cannot share resources (such as base model instance) between inference and fine-tuning jobs. 4) They do not provide privacy to users who may not wish to expose their fine-tuned parameters to service providers. In Symbiosis, we address the above problems by enabling as-a-service deployment of base model. The base model layers can be shared across multiple inference or fine-tuning processes. Our split-execution technique decouples the execution of client-specific adapters and layers from the frozen base model layers offering them flexibility to manage their resources, to select their fine-tuning method, to achieve their performance goals. Our approach is transparent to models and works out-of-the-box for most models in the transformers library. Our evaluation on Llama2-13B shows the compared to baseline, Symbiosis can fine-tune 4X more adapters on the same set of GPUs in the same amount of time. 

**Abstract (ZH)**: Parameter-efficient Fine-tuning with Shared Base Models in Symbiosis 

---
# Disclosing Generative AI Use in Digital Humanities Research 

**Title (ZH)**: 披露数字人文研究中生成式AI的应用 

**Authors**: Rongqian Ma, Xuhan Zhang, Adrian Wisnicki  

**Link**: [PDF](https://arxiv.org/pdf/2507.03216)  

**Abstract**: This survey study investigates how digital humanists perceive and approach generative AI disclosure in research. The results indicate that while digital humanities scholars acknowledge the importance of disclosing GenAI use, the actual rate of disclosure in research practice remains low. Respondents differ in their views on which activities most require disclosure and on the most appropriate methods for doing so. Most also believe that safeguards for AI disclosure should be established through institutional policies rather than left to individual decisions. The study's findings will offer empirical guidance to scholars, institutional leaders, funders, and other stakeholders responsible for shaping effective disclosure policies. 

**Abstract (ZH)**: 这项调查研究探讨了数字人文学家在研究中如何看待和处理生成式AI披露的问题。结果显示，虽然数字人文学者认为披露生成式AI使用的重要性，但在实际研究实践中，披露率仍然较低。受访者在哪些活动最需要披露以及应该如何披露方面观点不一。大多数研究人员认为，应该通过机构政策来制定AI披露的安全措施，而不是依赖个人决策。研究发现将为学者、机构领导者、资助者及其他负责制定有效披露政策的相关利益方提供实证指导。 

---
# AI-driven Web Application for Early Detection of Sudden Death Syndrome (SDS) in Soybean Leaves Using Hyperspectral Images and Genetic Algorithm 

**Title (ZH)**: 基于AI的高光谱图像和遗传算法在大豆叶片突发死亡综合症早期检测中的Web应用 

**Authors**: Pappu Kumar Yadav, Rishik Aggarwal, Supriya Paudel, Amee Parmar, Hasan Mirzakhaninafchi, Zain Ul Abideen Usmani, Dhe Yeong Tchalla, Shyam Solanki, Ravi Mural, Sachin Sharma, Thomas F. Burks, Jianwei Qin, Moon S. Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.03198)  

**Abstract**: Sudden Death Syndrome (SDS), caused by Fusarium virguliforme, poses a significant threat to soybean production. This study presents an AI-driven web application for early detection of SDS on soybean leaves using hyperspectral imaging, enabling diagnosis prior to visible symptom onset. Leaf samples from healthy and inoculated plants were scanned using a portable hyperspectral imaging system (398-1011 nm), and a Genetic Algorithm was employed to select five informative wavelengths (505.4, 563.7, 712.2, 812.9, and 908.4 nm) critical for discriminating infection status. These selected bands were fed into a lightweight Convolutional Neural Network (CNN) to extract spatial-spectral features, which were subsequently classified using ten classical machine learning models. Ensemble classifiers (Random Forest, AdaBoost), Linear SVM, and Neural Net achieved the highest accuracy (>98%) and minimal error across all folds, as confirmed by confusion matrices and cross-validation metrics. Poor performance by Gaussian Process and QDA highlighted their unsuitability for this dataset. The trained models were deployed within a web application that enables users to upload hyperspectral leaf images, visualize spectral profiles, and receive real-time classification results. This system supports rapid and accessible plant disease diagnostics, contributing to precision agriculture practices. Future work will expand the training dataset to encompass diverse genotypes, field conditions, and disease stages, and will extend the system for multiclass disease classification and broader crop applicability. 

**Abstract (ZH)**: 由Fusarium virguliforme引起的猝死综合症（SDS）对大豆生产构成了重大威胁。本研究提出了一种基于人工智能的网络应用，使用高光谱成像技术在可见症状出现之前对大豆叶片上的SDS进行早期检测，从而实现早期诊断。健康植物和感染植物的叶片样本通过便携式高光谱成像系统（398-1011 nm）进行扫描，并使用遗传算法选择了五个关键的波长 bands（505.4, 563.7, 712.2, 812.9, 和 908.4 nm）用于区分感染状态。这些选定的波长输入轻量级卷积神经网络（CNN）以提取空同谱特征，随后使用十种经典机器学习模型进行分类。集成分类器（随机森林、AdaBoost）、线性SVM和神经网络在所有折中取得了最高准确率（>98%）并具有最小误差，这得到了混淆矩阵和交叉验证指标的确认。高斯过程和支持向量机二次判别分析的较差性能表明它们不适于该数据集。训练好的模型部署在web应用中，使用户能够上传高光谱叶片图像、可视化光谱特征并获得实时分类结果。该系统支持快速和便捷的植物疾病诊断，助力精准农业实践。未来工作将扩大训练数据集以涵盖多样化的基因型、田间条件和疾病阶段，并将系统扩展到多类疾病分类及更广泛的大田作物应用。 

---
# How Much Content Do LLMs Generate That Induces Cognitive Bias in Users? 

**Title (ZH)**: LLMs生成的诱导用户认知偏见的内容有多少？ 

**Authors**: Abeer Alessa, Akshaya Lakshminarasimhan, Param Somane, Julian Skirzynski, Julian McAuley, Jessica Echterhoff  

**Link**: [PDF](https://arxiv.org/pdf/2507.03194)  

**Abstract**: Large language models (LLMs) are increasingly integrated into applications ranging from review summarization to medical diagnosis support, where they affect human decisions. Even though LLMs perform well in many tasks, they may also inherit societal or cognitive biases, which can inadvertently transfer to humans. We investigate when and how LLMs expose users to biased content and quantify its severity. Specifically, we assess three LLM families in summarization and news fact-checking tasks, evaluating how much LLMs stay consistent with their context and/or hallucinate. Our findings show that LLMs expose users to content that changes the sentiment of the context in 21.86% of the cases, hallucinates on post-knowledge-cutoff data questions in 57.33% of the cases, and primacy bias in 5.94% of the cases. We evaluate 18 distinct mitigation methods across three LLM families and find that targeted interventions can be effective. Given the prevalent use of LLMs in high-stakes domains, such as healthcare or legal analysis, our results highlight the need for robust technical safeguards and for developing user-centered interventions that address LLM limitations. 

**Abstract (ZH)**: 大型语言模型（LLMs）在从评论摘要到医疗诊断支持的应用中日益集成，影响人类决策。尽管LLMs在许多任务中表现良好，但也可能继承社会或认知偏见，这些偏见可能无意间转移到人类身上。我们研究了LLMs在摘要和新闻事实核查任务中何时以及如何向用户暴露偏见内容，并量化其严重程度。具体而言，我们在三个LLM家族中评估了其在摘要和新闻事实核查任务中的表现，评估LLMs在多大程度上保持一致性或胡言乱语。我们的研究表明，在21.86%的情况下，LLMs使上下文的情感发生变化，在57.33%的情况下对后知识截止日期的数据问题进行胡言乱语，在5.94%的情况下表现出首因效应偏见。我们在三个LLM家族中评估了18种不同的缓解方法，发现针对性的干预措施可能有效。鉴于LLMs在高风险领域，如医疗保健或法律分析中的广泛应用，我们的研究结果强调了需要 robust的技术保障，并开发以用户为中心的干预措施来解决LLM的局限性。 

---
# Deep Learning Atmospheric Models Reliably Simulate Out-of-Sample Land Heat and Cold Wave Frequencies 

**Title (ZH)**: 深度学习大气模型可靠模拟样本外土地热浪和冷浪频率 

**Authors**: Zilu Meng, Gregory J. Hakim, Wenchang Yang, Gabriel A. Vecchi  

**Link**: [PDF](https://arxiv.org/pdf/2507.03176)  

**Abstract**: Deep learning (DL)-based general circulation models (GCMs) are emerging as fast simulators, yet their ability to replicate extreme events outside their training range remains unknown. Here, we evaluate two such models -- the hybrid Neural General Circulation Model (NGCM) and purely data-driven Deep Learning Earth System Model (DL\textit{ESy}M) -- against a conventional high-resolution land-atmosphere model (HiRAM) in simulating land heatwaves and coldwaves. All models are forced with observed sea surface temperatures and sea ice over 1900-2020, focusing on the out-of-sample early-20th-century period (1900-1960). Both DL models generalize successfully to unseen climate conditions, broadly reproducing the frequency and spatial patterns of heatwave and cold wave events during 1900-1960 with skill comparable to HiRAM. An exception is over portions of North Asia and North America, where all models perform poorly during 1940-1960. Due to excessive temperature autocorrelation, DL\textit{ESy}M tends to overestimate heatwave and cold wave frequencies, whereas the physics-DL hybrid NGCM exhibits persistence more similar to HiRAM. 

**Abstract (ZH)**: 基于深度学习的通用环流模型在模拟土地热浪和冷锋方面的能力评价 

---
# Understanding Knowledge Transferability for Transfer Learning: A Survey 

**Title (ZH)**: 理解迁移学习中的知识可迁移性：一个综述 

**Authors**: Haohua Wang, Jingge Wang, Zijie Zhao, Yang Tan, Yanru Wu, Hanbing Liu, Jingyun Yang, Enming Zhang, Xiangyu Chen, Zhengze Rong, Shanxin Guo, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.03175)  

**Abstract**: Transfer learning has become an essential paradigm in artificial intelligence, enabling the transfer of knowledge from a source task to improve performance on a target task. This approach, particularly through techniques such as pretraining and fine-tuning, has seen significant success in fields like computer vision and natural language processing. However, despite its widespread use, how to reliably assess the transferability of knowledge remains a challenge. Understanding the theoretical underpinnings of each transferability metric is critical for ensuring the success of transfer learning. In this survey, we provide a unified taxonomy of transferability metrics, categorizing them based on transferable knowledge types and measurement granularity. This work examines the various metrics developed to evaluate the potential of source knowledge for transfer learning and their applicability across different learning paradigms emphasizing the need for careful selection of these metrics. By offering insights into how different metrics work under varying conditions, this survey aims to guide researchers and practitioners in selecting the most appropriate metric for specific applications, contributing to more efficient, reliable, and trustworthy AI systems. Finally, we discuss some open challenges in this field and propose future research directions to further advance the application of transferability metrics in trustworthy transfer learning. 

**Abstract (ZH)**: 迁移学习已成为人工智能中的一个基本范式，通过将源自一个任务的知识转移到另一个目标任务以提高其性能。尽管这种方法在计算机视觉和自然语言处理等领域取得了显著成功，但如何可靠地评估知识的可迁移性仍是一个挑战。了解每个迁移性度量的理论基础对于确保迁移学习的成功至关重要。在这篇综述中，我们提供了一个统一的迁移性度量分类框架，根据可迁移知识类型和测量粒度对其进行分类。本文考查了不同的度量方法，以评估源知识在迁移学习中的潜在能力及其在不同学习范式中的适用性，强调了谨慎选择这些度量方法的必要性。通过揭示不同度量方法在不同条件下的工作机制，本文旨在指导研究人员和从业人员选择最适合特定应用的度量方法，从而促进更高效、可靠和可信的人工智能系统的发展。最后，我们讨论了该领域的一些开放挑战，并提出了未来研究方向，旨在进一步推动可信赖迁移学习中迁移性度量的应用。 

---
# Adversarial Manipulation of Reasoning Models using Internal Representations 

**Title (ZH)**: 利用内部表示操纵推理模型的对抗性攻击 

**Authors**: Kureha Yamaguchi, Benjamin Etheridge, Andy Arditi  

**Link**: [PDF](https://arxiv.org/pdf/2507.03167)  

**Abstract**: Reasoning models generate chain-of-thought (CoT) tokens before their final output, but how this affects their vulnerability to jailbreak attacks remains unclear. While traditional language models make refusal decisions at the prompt-response boundary, we find evidence that DeepSeek-R1-Distill-Llama-8B makes these decisions within its CoT generation. We identify a linear direction in activation space during CoT token generation that predicts whether the model will refuse or comply -- termed the "caution" direction because it corresponds to cautious reasoning patterns in the generated text. Ablating this direction from model activations increases harmful compliance, effectively jailbreaking the model. We additionally show that intervening only on CoT token activations suffices to control final outputs, and that incorporating this direction into prompt-based attacks improves success rates. Our findings suggest that the chain-of-thought itself is a promising new target for adversarial manipulation in reasoning models.
Code available at this https URL 

**Abstract (ZH)**: 基于链-of- thought生成的推理模型在生成最终输出之前会生成链-of-thought（CoT）令牌，但这种过程如何影响其对 jailbreak 攻击的脆弱性尚不明确。虽然传统的语言模型在提示-响应边界处做出拒绝决策，但我们发现 DeepSeek-R1-Distill-Llama-8B 在生成CoT时在其推理过程中也会做出这些决策。我们发现在CoT令牌生成过程中激活空间中存在一条线性方向，该方向可以预测模型是否会拒绝或遵从，称为“谨慎”方向，因为它对应于生成文本中的谨慎推理模式。从模型激活中去除这一方向会增加有害遵从，有效地使模型 jailbreak。此外，我们证明仅干预CoT令牌激活即可控制最终输出，并且将此方向纳入基于提示的攻击可以提高成功几率。我们的研究结果表明，链-of-thought本身就是推理模型对抗操纵的一个有希望的新靶标。 

---
# MateInfoUB: A Real-World Benchmark for Testing LLMs in Competitive, Multilingual, and Multimodal Educational Tasks 

**Title (ZH)**: MateInfoUB：用于测试在竞争性、多语言和多模态教育任务中大型语言模型的现实世界基准 

**Authors**: Dumitran Adrian Marius, Theodor-Pierre Moroianu, Buca Mihnea-Vicentiu  

**Link**: [PDF](https://arxiv.org/pdf/2507.03162)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has transformed various domains, particularly computer science (CS) education. These models exhibit remarkable capabilities in code-related tasks and problem-solving, raising questions about their potential and limitations in advanced CS contexts. This study presents a novel bilingual (English-Romanian) multimodal (text and image) dataset of multiple-choice questions derived from a high-level computer science competition. A particularity of our dataset is that the problems are conceived such that some of them are easier solved using reasoning on paper, while for others writing code is more efficient. We systematically evaluate State of The Art LLMs on this dataset, analyzing their performance on theoretical programming tasks. Our findings reveal the strengths and limitations of current LLMs, including the influence of language choice (English vs. Romanian), providing insights into their applicability in CS education and competition settings. We also address critical ethical considerations surrounding educational integrity and the fairness of assessments in the context of LLM usage. These discussions aim to inform future educational practices and policies. To support further research, our dataset will be made publicly available in both English and Romanian. Additionally, we release an educational application tailored for Romanian students, enabling them to self-assess using the dataset in an interactive and practice-oriented environment. 

**Abstract (ZH)**: 大型语言模型的快速进步已transformed various领域，尤其是在计算机科学（CS）教育方面。这些模型在代码相关任务和问题解决方面展现出非凡的能力，引发了它们在高级CS环境中的潜力和局限性的思考。本研究提出了一种新颖的双语（英语-罗曼语）多模态（文本和图像）试题集，源自一项高级计算机科学竞赛。该数据集的一个特点是，部分问题更适合通过纸上推理解决，而对于其他问题，则编写代码更为高效。我们系统地评估了当前最先进的大型语言模型在该数据集上的表现，分析了它们在理论编程任务中的性能。我们的研究发现揭示了当前大型语言模型的优势和局限性，包括语言选择（英语 vs. 罗曼语）的影响，提供了它们在CS教育和竞赛环境中的应用见解。此外，我们还探讨了大型语言模型使用背景下教育诚信和评估公平性的关键伦理考虑。这些讨论旨在指导未来的教育实践和政策。为了支持进一步的研究，该数据集将以英语和罗曼语形式公开发布。此外，我们还发布了一款针对罗曼语学生定制的教育应用，使他们能够在交互性和实践导向的环境中自我评估。 

---
# The Impact of LLM-Assistants on Software Developer Productivity: A Systematic Literature Review 

**Title (ZH)**: LLM助手对软件开发者生产力的影响：一项系统文献综述 

**Authors**: Amr Mohamed, Maram Assi, Mariam Guizani  

**Link**: [PDF](https://arxiv.org/pdf/2507.03156)  

**Abstract**: Large language model assistants (LLM-assistants) present new opportunities to transform software development. Developers are increasingly adopting these tools across tasks, including coding, testing, debugging, documentation, and design. Yet, despite growing interest, there is no synthesis of how LLM-assistants affect software developer productivity. In this paper, we present a systematic literature review of 37 peer-reviewed studies published between January 2014 and December 2024 that examine this impact. Our analysis reveals that LLM-assistants offer both considerable benefits and critical risks. Commonly reported gains include minimized code search, accelerated development, and the automation of trivial and repetitive tasks. However, studies also highlight concerns around cognitive offloading, reduced team collaboration, and inconsistent effects on code quality. While the majority of studies (92%) adopt a multi-dimensional perspective by examining at least two SPACE dimensions, reflecting increased awareness of the complexity of developer productivity, only 14% extend beyond three dimensions, indicating substantial room for more integrated evaluations. Satisfaction, Performance, and Efficiency are the most frequently investigated dimensions, whereas Communication and Activity remain underexplored. Most studies are exploratory (64%) and methodologically diverse, but lack longitudinal and team-based evaluations. This review surfaces key research gaps and provides recommendations for future research and practice. All artifacts associated with this study are publicly available at this https URL. 

**Abstract (ZH)**: 大型语言模型助手（LLM助手）为软件开发转型提供了新机遇。开发人员越来越多地在编码、测试、调试、文档编写和设计等任务中采用这些工具。然而，尽管兴趣日益浓厚，却没有综合分析LLM助手如何影响软件开发者生产力。在本文中，我们对2014年1月到2024年12月间发表的37篇同行评审研究进行了系统文献综述，这些研究探讨了这一影响。我们的分析表明，LLM助手提供了显著的利益，同时也带来了关键的风险。常见的获益包括代码搜索减少、开发加速以及自动化的琐碎和重复性任务。然而，研究也指出了认知卸载、团队协作减少以及代码质量不一致等关切。虽然大部分研究（92%）采用了多维度视角，至少考察了两个SPACE维度，反映了对开发者生产力复杂性的更强认知，但仅有14%的研究扩展到超过三个维度，表明对更全面评估仍有很大空间。满意度、绩效和效率是研究中最为频繁考察的维度，而沟通和活动则仍然被研究较少。大多数研究具有探索性（64%）且方法论多样化，但缺乏纵向和基于团队的评估。本综述揭示了关键研究空白，并为未来的研究和实践提供了建议。与本研究相关的所有成果均可通过此 https URL 公开获取。 

---
# Expert-level validation of AI-generated medical text with scalable language models 

**Title (ZH)**: 基于可扩展语言模型的AI生成医疗文本专家级验证 

**Authors**: Asad Aali, Vasiliki Bikia, Maya Varma, Nicole Chiou, Sophie Ostmeier, Arnav Singhvi, Magdalini Paschali, Ashwin Kumar, Andrew Johnston, Karimar Amador-Martinez, Eduardo Juan Perez Guerrero, Paola Naovi Cruz Rivera, Sergios Gatidis, Christian Bluethgen, Eduardo Pontes Reis, Eddy D. Zandee van Rilland, Poonam Laxmappa Hosamani, Kevin R Keet, Minjoung Go, Evelyn Ling, David B. Larson, Curtis Langlotz, Roxana Daneshjou, Jason Hom, Sanmi Koyejo, Emily Alsentzer, Akshay S. Chaudhari  

**Link**: [PDF](https://arxiv.org/pdf/2507.03152)  

**Abstract**: With the growing use of language models (LMs) in clinical environments, there is an immediate need to evaluate the accuracy and safety of LM-generated medical text. Currently, such evaluation relies solely on manual physician review. However, detecting errors in LM-generated text is challenging because 1) manual review is costly and 2) expert-composed reference outputs are often unavailable in real-world settings. While the "LM-as-judge" paradigm (a LM evaluating another LM) offers scalable evaluation, even frontier LMs can miss subtle but clinically significant errors. To address these challenges, we propose MedVAL, a self-supervised framework that leverages synthetic data to train evaluator LMs to assess whether LM-generated medical outputs are factually consistent with inputs, without requiring physician labels or reference outputs. To evaluate LM performance, we introduce MedVAL-Bench, a dataset containing 840 outputs annotated by physicians, following a physician-defined taxonomy of risk levels and error categories. Across 6 diverse medical tasks and 10 state-of-the-art LMs spanning open-source, proprietary, and medically adapted models, MedVAL fine-tuning significantly improves (p < 0.001) alignment with physicians on both seen and unseen tasks, increasing average F1 scores from 66% to 83%, with per-sample safety classification scores up to 86%. MedVAL improves the performance of even the best-performing proprietary LM (GPT-4o) by 8%. To support a scalable, risk-aware pathway towards clinical integration, we open-source the 1) codebase ( this https URL ), 2) MedVAL-Bench ( this https URL ), and 3) MedVAL-4B ( this https URL ), the best-performing open-source LM. Our research provides the first evidence of LMs approaching expert-level validation ability for medical text. 

**Abstract (ZH)**: 利用合成数据训练自我监督框架以评估语言模型生成的医学文本准确性与安全性：MedVAL框架 

---
# On the Relationship between Accent Strength and Articulatory Features 

**Title (ZH)**: 重音强度与发音特征之间的关系 

**Authors**: Kevin Huang, Sean Foley, Jihwan Lee, Yoonjeong Lee, Dani Byrd, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2507.03149)  

**Abstract**: This paper explores the relationship between accent strength and articulatory features inferred from acoustic speech. To quantify accent strength, we compare phonetic transcriptions with transcriptions based on dictionary-based references, computing phoneme-level difference as a measure of accent strength. The proposed framework leverages recent self-supervised learning articulatory inversion techniques to estimate articulatory features. Analyzing a corpus of read speech from American and British English speakers, this study examines correlations between derived articulatory parameters and accent strength proxies, associating systematic articulatory differences with indexed accent strength. Results indicate that tongue positioning patterns distinguish the two dialects, with notable differences inter-dialects in rhotic and low back vowels. These findings contribute to automated accent analysis and articulatory modeling for speech processing applications. 

**Abstract (ZH)**: 本文探索了音重强度与从语音声学中推断出的articulatory特征之间的关系。为了量化音重强度，我们将语音转录与基于词典的参考转录进行比较，计算音素级差异作为音重强度的度量。本文提出的框架利用了近期的自监督学习articulatory反转技术来估计articulatory特征。通过对来自美国和英国英语演讲者的朗读语音语料库的分析，本研究探讨了导出的articulatory参数与音重强度代理变量之间的相关性，将系统的articulatory差异与索引的音重强度关联起来。结果表明，舌尖定位模式区分了这两种方言，音重音和低后元音在方言之间表现出显著差异。这些发现为语音处理应用中的自动化音重分析和articulatory建模做出了贡献。 

---
# How Overconfidence in Initial Choices and Underconfidence Under Criticism Modulate Change of Mind in Large Language Models 

**Title (ZH)**: 初始选择中的过度自信与批评中的欠自信如何调Modulate大型语言模型中的改变认知 

**Authors**: Dharshan Kumaran, Stephen M Fleming, Larisa Markeeva, Joe Heyward, Andrea Banino, Mrinal Mathur, Razvan Pascanu, Simon Osindero, Benedetto de Martino, Petar Velickovic, Viorica Patraucean  

**Link**: [PDF](https://arxiv.org/pdf/2507.03120)  

**Abstract**: Large language models (LLMs) exhibit strikingly conflicting behaviors: they can appear steadfastly overconfident in their initial answers whilst at the same time being prone to excessive doubt when challenged. To investigate this apparent paradox, we developed a novel experimental paradigm, exploiting the unique ability to obtain confidence estimates from LLMs without creating memory of their initial judgments -- something impossible in human participants. We show that LLMs -- Gemma 3, GPT4o and o1-preview -- exhibit a pronounced choice-supportive bias that reinforces and boosts their estimate of confidence in their answer, resulting in a marked resistance to change their mind. We further demonstrate that LLMs markedly overweight inconsistent compared to consistent advice, in a fashion that deviates qualitatively from normative Bayesian updating. Finally, we demonstrate that these two mechanisms -- a drive to maintain consistency with prior commitments and hypersensitivity to contradictory feedback -- parsimoniously capture LLM behavior in a different domain. Together, these findings furnish a mechanistic account of LLM confidence that explains both their stubbornness and excessive sensitivity to criticism. 

**Abstract (ZH)**: 大型语言模型（LLMs）表现出令人惊讶的矛盾行为：它们在初始答案上显得异常自信，而在受到质疑时却又容易过度怀疑。为了探究这一显而易见的悖论，我们开发了一个新颖的实验范式，利用了获取LLMs置信度估计的独特能力，而无需形成它们初始判断的记忆——这是人类参与者无法做到的。我们显示，LLMs——Gemma 3、GPT4o 和 o1-preview——表现出明显的选择支持偏见，这种偏见强化并提升了它们对答案的置信度估计，导致它们顽固地抵制改变观点。我们进一步证明，LLMs对不一致建议的权重明显高于一致建议，这种从量上偏离了规范化的贝叶斯更新。最后，我们证明，这两种机制——保持与先前承诺一致性的驱动力和对矛盾反馈的超敏感性——能够简明地捕捉LLMs在不同领域中的行为。这些发现为解释LLMs的顽固性和过度敏感性提供了机理性的解释。 

---
# Neural-Network solver of ideal MHD equilibria 

**Title (ZH)**: 理想的MHD等离子体平衡的神经网络求解器 

**Authors**: Timo Thun, Andrea Merlo, Rory Conlin, Dario Panici, Daniel Böckenhoff  

**Link**: [PDF](https://arxiv.org/pdf/2507.03119)  

**Abstract**: We present a novel approach to compute three-dimensional Magnetohydrodynamic equilibria by parametrizing Fourier modes with artificial neural networks and compare it to equilibria computed by conventional solvers. The full nonlinear global force residual across the volume in real space is then minimized with first order optimizers. Already,we observe competitive computational cost to arrive at the same minimum residuals computed by existing codes. With increased computational cost,lower minima of the residual are achieved by the neural networks,establishing a new lower bound for the force residual. We use minimally complex neural networks,and we expect significant improvements for solving not only single equilibria with neural networks,but also for computing neural network models valid over continuous distributions of equilibria. 

**Abstract (ZH)**: 我们提出了一种新的方法，通过使用人工神经网络参数化傅里叶模式来计算三维磁流体力学平衡，并将其与传统求解器计算的平衡进行比较。然后，在实空间中最小化体积上的完整非线性全局力残差，使用一阶优化器。我们观察到，与现有代码计算相同最小残差相比，已实现相近的计算成本。随着计算成本的增加，神经网络实现了更低的残差最小值，从而确立了力残差的新下限。我们使用了结构简单的神经网络，并期望通过神经网络不仅能够解决单个平衡问题，还能计算适用于连续平衡分布的神经网络模型，从而获得显著改进。 

---
# RLVER: Reinforcement Learning with Verifiable Emotion Rewards for Empathetic Agents 

**Title (ZH)**: RLVER：可验证情绪reward的强化学习方法用于 empathy代理 

**Authors**: Peisong Wang, Ruotian Ma, Bang Zhang, Xingyu Chen, Zhiwei He, Kang Luo, Qingsong Lv, Qingxuan Jiang, Zheng Xie, Shanyi Wang, Yuan Li, Fanghua Ye, Jian Li, Yifan Yang, Zhaopeng Tu, Xiaolong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.03112)  

**Abstract**: Large language models (LLMs) excel at logical and algorithmic reasoning, yet their emotional intelligence (EQ) still lags far behind their cognitive prowess. While reinforcement learning from verifiable rewards (RLVR) has advanced in other domains, its application to dialogue-especially for emotional intelligence-remains underexplored. In this work, we introduce RLVER, the first end-to-end reinforcement learning framework that leverages verifiable emotion rewards from simulated users to cultivate higher-order empathetic abilities in LLMs. Within this framework, self-consistent affective simulated users engage in dialogue rollouts and produce deterministic emotion scores during conversations, serving as reward signals to guide the LLM's learning. Fine-tuning publicly available Qwen2.5-7B-Instruct model with PPO boosts its Sentient-Benchmark score from 13.3 to 79.2 while largely preserving mathematical and coding competence. Extensive experiments reveal that: (i) RLVER consistently improves multiple dialogue capabilities; (ii) Thinking and non-thinking models show distinct trends--thinking models excel in empathy and insight, while non-thinking models favor action; (iii) GRPO often yields stable gains, while PPO can push certain capabilities to a higher ceiling; (iv) More challenging environments are not always better-moderate ones can yield stronger outcomes. Our results show that RLVER is a practical route toward emotionally intelligent and broadly capable language agents. 

**Abstract (ZH)**: 大型语言模型在逻辑和算法推理方面表现卓越，但在情感 intelligence（EQ）方面仍远远落后于其认知能力。尽管可验证奖励强化学习（RLVR）在其他领域取得了进展，但在对话中特别是情感 intelligence 方面的应用仍待探索。在这项工作中，我们引入了 RLVER，这是第一个利用模拟用户的情感验证奖励来培养大型语言模型高阶同理能力的端到端强化学习框架。在此框架中，自我一致的情感模拟用户参与对话展开，并在会话中生成确定性的情感得分，作为奖励信号引导大型语言模型的学习。使用 PPO 算法 fine-tune 公开可用的 Qwen2.5-7B-Instruct 模型，使其 Sentient-Benchmark 得分从 13.3 提升到 79.2，同时主要保留了数学和编程能力。广泛的实验表明：（i）RLVER 一致地提高多种对话能力；（ii）思考模型和非思考模型表现出不同的趋势——思考模型在同理心和洞察力方面表现出色，而非思考模型则更侧重于行动；（iii）GRPO 经常产生稳定的收益，而 PPO 可以推动某些能力达到更高的上限；（iv）更具挑战性的环境并不总是更好的选择——适度的环境可能产生更强的效果。我们的结果表明，RLVER 是实现具备情感 intelligence 和广泛能力的语言代理的现实途径。 

---
# Uncovering Synergistic Educational Injustices of COVID-19 and AI 

**Title (ZH)**: 揭示COVID-19与人工智能协同的教育不公 

**Authors**: Ahmad Banyasady  

**Link**: [PDF](https://arxiv.org/pdf/2507.03095)  

**Abstract**: Grounded in critical realism and using narrative inquiry, this article explores this article explores the long-term consequences of the COVID-19 pandemic and the rapid proliferation of artificial intelligence within higher education. Through the analysis of student narratives collected in Iranian university settings, the study reveals that learning experiences during and after the pandemic, coupled with unprepared exposure to AI tools, have generated hidden yet impactful layers of educational inequality and cognitive disorientation. 

**Abstract (ZH)**: 基于批判现实主义和叙事研究的方法，本文探讨了COVID-19 pandemic和人工智能在高等教育领域快速普及的长期后果。通过分析在伊朗大学环境中收集的学生叙事，研究揭示了在疫情期间及之后的学习经历，以及对AI工具的未准备充分的暴露，已经产生了隐藏但具有影响力的教育不平等和认知错位的层面。 

---
# ARF-RLHF: Adaptive Reward-Following for RLHF through Emotion-Driven Self-Supervision and Trace-Biased Dynamic Optimization 

**Title (ZH)**: 基于情绪驱动自监督和轨迹偏差动态优化的自适应奖励跟随：用于RLHF的过程调整与优化 

**Authors**: YuXuan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03069)  

**Abstract**: With the rapid advancement of Reinforcement Learning from Human Feedback (RLHF) and autoregressive transformers, state-of-the-art models such as GPT-4.0, DeepSeek R1, and Llama 3.3 increasingly emphasize answer depth and personalization. However, most existing RLHF approaches (e.g., PPO, DPO) still rely on a binary-preference (BT) paradigm, which, while reducing annotation costs, still requires substantial human effort and captures only group-level tendencies rather than individual preferences. To overcome these limitations, we propose Adaptive Reward-Following (ARF), a self-assessment framework that leverages a high-precision emotion analyzer achieving over 70% accuracy on GoEmotions, Sentiment140, and DailyDialog to convert free-form user feedback into continuous preference scores. We further enrich and debias these signals through lightweight data augmentations, including synonym replacement, random trace truncation, and score bias annotation algorithm. A Dynamic Adapter Preference Tracker continuously models evolving user tastes in real time, enabling our novel Trace Bias (TB) fine-tuning algorithm to optimize directly on these tracked rewards instead of coarse binary labels. Experiments on Qwen-2/2.5, Gemma-2, and Llama-3.2 across four preference domains demonstrate that ARF achieves an improvement of 3.3% over PPO and 7.6% over DPO. Moreover, TB preserves theoretical alignment with PPO and DPO objectives. Overall, ARF presents a scalable, personalized, and cost-effective approach to RLHF LLMs through autonomous reward modeling. 

**Abstract (ZH)**: 基于自评估的自适应奖励跟随（ARF）：强化学习从人类反馈的个性化模型设计 

---
# Large Language Models for Automating Clinical Data Standardization: HL7 FHIR Use Case 

**Title (ZH)**: 大型语言模型在自动化临床数据标准化中的应用：HL7 FHIR案例研究 

**Authors**: Alvaro Riquelme, Pedro Costa, Catalina Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2507.03067)  

**Abstract**: For years, semantic interoperability standards have sought to streamline the exchange of clinical data, yet their deployment remains time-consuming, resource-intensive, and technically challenging. To address this, we introduce a semi-automated approach that leverages large language models specifically GPT-4o and Llama 3.2 405b to convert structured clinical datasets into HL7 FHIR format while assessing accuracy, reliability, and security. Applying our method to the MIMIC-IV database, we combined embedding techniques, clustering algorithms, and semantic retrieval to craft prompts that guide the models in mapping each tabular field to its corresponding FHIR resource. In an initial benchmark, resource identification achieved a perfect F1-score, with GPT-4o outperforming Llama 3.2 thanks to the inclusion of FHIR resource schemas within the prompt. Under real-world conditions, accuracy dipped slightly to 94 %, but refinements to the prompting strategy restored robust mappings. Error analysis revealed occasional hallucinations of non-existent attributes and mismatches in granularity, which more detailed prompts can mitigate. Overall, our study demonstrates the feasibility of context-aware, LLM-driven transformation of clinical data into HL7 FHIR, laying the groundwork for semi-automated interoperability workflows. Future work will focus on fine-tuning models with specialized medical corpora, extending support to additional standards such as HL7 CDA and OMOP, and developing an interactive interface to enable expert validation and iterative refinement. 

**Abstract (ZH)**: 基于大规模语言模型的半自动化临床数据转换为HL7 FHIR格式的方法研究 

---
# Identification of Potentially Misclassified Crash Narratives using Machine Learning (ML) and Deep Learning (DL) 

**Title (ZH)**: 使用机器学习（ML）和深度学习（DL）识别潜在分类错误的道路交通事故叙述 

**Authors**: Sudesh Bhagat, Ibne Farabi Shihab, Jonathan Wood  

**Link**: [PDF](https://arxiv.org/pdf/2507.03066)  

**Abstract**: This research investigates the efficacy of machine learning (ML) and deep learning (DL) methods in detecting misclassified intersection-related crashes in police-reported narratives. Using 2019 crash data from the Iowa Department of Transportation, we implemented and compared a comprehensive set of models, including Support Vector Machine (SVM), XGBoost, BERT Sentence Embeddings, BERT Word Embeddings, and Albert Model. Model performance was systematically validated against expert reviews of potentially misclassified narratives, providing a rigorous assessment of classification accuracy. Results demonstrated that while traditional ML methods exhibited superior overall performance compared to some DL approaches, the Albert Model achieved the highest agreement with expert classifications (73% with Expert 1) and original tabular data (58%). Statistical analysis revealed that the Albert Model maintained performance levels similar to inter-expert consistency rates, significantly outperforming other approaches, particularly on ambiguous narratives. This work addresses a critical gap in transportation safety research through multi-modal integration analysis, which achieved a 54.2% reduction in error rates by combining narrative text with structured crash data. We conclude that hybrid approaches combining automated classification with targeted expert review offer a practical methodology for improving crash data quality, with substantial implications for transportation safety management and policy development. 

**Abstract (ZH)**: 本研究探讨了机器学习（ML）和深度学习（DL）方法在检测警察报告中误分类的交叉口相关事故中的有效性。利用2019年爱达荷州交通运输部的事故数据，我们实施并比较了一整套模型，包括支持向量机（SVM）、XGBoost、BERT句子嵌入、BERT单词嵌入和Albert模型。模型性能系统地与潜在误分类的报告的专家评审结果进行了验证，提供了分类准确性的严格评估。结果表明，尽管传统机器学习方法在总体性能上优于某些深度学习方法，但Albert模型与专家分类的一致性最高（与专家1的一致性为73%，与原始表格数据的一致性为58%）。统计分析表明，Albert模型的性能水平类似于专家间的一致性率，显著优于其他方法，特别是在模糊的报告文本方面。通过将报告文本与结构化事故数据结合进行多模态综合分析，本研究填补了交通安全研究中的一个重要空白，实现了错误率减少了54.2%。我们得出结论，结合自动分类与目标化专家审阅的混合方法为提高事故数据质量提供了一种实用的方法，对交通安全管理与政策制定具有重大影响。 

---
# LLM-Driven Auto Configuration for Transient IoT Device Collaboration 

**Title (ZH)**: 由LLM驱动的临时物联网设备协作自动配置 

**Authors**: Hetvi Shastri, Walid A. Hanafy, Li Wu, David Irwin, Mani Srivastava, Prashant Shenoy  

**Link**: [PDF](https://arxiv.org/pdf/2507.03064)  

**Abstract**: Today's Internet of Things (IoT) has evolved from simple sensing and actuation devices to those with embedded processing and intelligent services, enabling rich collaborations between users and their devices. However, enabling such collaboration becomes challenging when transient devices need to interact with host devices in temporarily visited environments. In such cases, fine-grained access control policies are necessary to ensure secure interactions; however, manually implementing them is often impractical for non-expert users. Moreover, at run-time, the system must automatically configure the devices and enforce such fine-grained access control rules. Additionally, the system must address the heterogeneity of devices.
In this paper, we present CollabIoT, a system that enables secure and seamless device collaboration in transient IoT environments. CollabIoT employs a Large language Model (LLM)-driven approach to convert users' high-level intents to fine-grained access control policies. To support secure and seamless device collaboration, CollabIoT adopts capability-based access control for authorization and uses lightweight proxies for policy enforcement, providing hardware-independent abstractions.
We implement a prototype of CollabIoT's policy generation and auto configuration pipelines and evaluate its efficacy on an IoT testbed and in large-scale emulated environments. We show that our LLM-based policy generation pipeline is able to generate functional and correct policies with 100% accuracy. At runtime, our evaluation shows that our system configures new devices in ~150 ms, and our proxy-based data plane incurs network overheads of up to 2 ms and access control overheads up to 0.3 ms. 

**Abstract (ZH)**: 今岁的物联网（IoT）已从简单的传感和执行设备演变为主-cols上嵌入处理和智能服务的设备，使得用户与其设备之间能够进行丰富的协作。然而，当临时设备需要在临时访问的环境中与宿主设备交互时，实现这种协作变得具有挑战性。在这种情况下，需要精细的访问控制策略来确保安全的交互；但是，非专家用户手动实现这些策略往往是不切实际的。此外，在运行时，系统必须自动配置设备并强制执行这样的精细访问控制规则。此外，该系统必须解决设备的异构性。
本文介绍了CollabIoT系统，该系统能够使临时物联网环境中设备的协作安全且无缝。CollabIoT采用了基于大型语言模型（LLM）的方法，将用户的高层次意图转换为精细的访问控制策略。为了支持安全且无缝的设备协作，CollabIoT采用了基于能力的访问控制用于授权，并使用轻量级代理进行策略执行，提供硬件无关的抽象。 

---
# BERT4Traj: Transformer Based Trajectory Reconstruction for Sparse Mobility Data 

**Title (ZH)**: BERT4Traj：基于Transformer的稀疏移动数据轨迹重建 

**Authors**: Hao Yang, Angela Yao, Christopher Whalen, Gengchen Mai  

**Link**: [PDF](https://arxiv.org/pdf/2507.03062)  

**Abstract**: Understanding human mobility is essential for applications in public health, transportation, and urban planning. However, mobility data often suffers from sparsity due to limitations in data collection methods, such as infrequent GPS sampling or call detail record (CDR) data that only capture locations during communication events. To address this challenge, we propose BERT4Traj, a transformer based model that reconstructs complete mobility trajectories by predicting hidden visits in sparse movement sequences. Inspired by BERT's masked language modeling objective and self_attention mechanisms, BERT4Traj leverages spatial embeddings, temporal embeddings, and contextual background features such as demographics and anchor points. We evaluate BERT4Traj on real world CDR and GPS datasets collected in Kampala, Uganda, demonstrating that our approach significantly outperforms traditional models such as Markov Chains, KNN, RNNs, and LSTMs. Our results show that BERT4Traj effectively reconstructs detailed and continuous mobility trajectories, enhancing insights into human movement patterns. 

**Abstract (ZH)**: 理解人类移动模式对于公共卫生、交通运输和城市规划等领域具有重要意义。然而，移动数据常常由于数据收集方法的限制（如不频繁的GPS采样或仅在通信事件中捕获位置的呼叫详细记录数据）而出现稀疏性。为解决这一挑战，我们提出了一种基于转换器的模型BERT4Traj，通过预测稀疏移动序列中的隐藏访问来重建完整的移动轨迹。受BERT的掩码语言建模目标和自注意力机制的启发，BERT4Traj利用空间嵌入、时间嵌入以及人口统计学特征和锚点等背景上下文信息。我们在乌干达坎帕拉收集的真实世界呼叫详细记录数据和GPS数据上评估了BERT4Traj，结果显示我们的方法显著优于Markov链、KNN、RNN和LSTM等传统模型。我们的结果表明，BERT4Traj有效地重建了详细且连续的移动轨迹，增强了对人类移动模式的见解。 

---
# AI-Based Reconstruction from Inherited Personal Data: Analysis, Feasibility, and Prospects 

**Title (ZH)**: 基于遗传个人数据的AI重构：分析、可行性和前景 

**Authors**: Mark Zilberman  

**Link**: [PDF](https://arxiv.org/pdf/2507.03059)  

**Abstract**: This article explores the feasibility of creating an "electronic copy" of a deceased researcher by training artificial intelligence (AI) on the data stored in their personal computers. By analyzing typical data volumes on inherited researcher computers, including textual files such as articles, emails, and drafts, it is estimated that approximately one million words are available for AI training. This volume is sufficient for fine-tuning advanced pre-trained models like GPT-4 to replicate a researcher's writing style, domain expertise, and rhetorical voice with high fidelity. The study also discusses the potential enhancements from including non-textual data and file metadata to enrich the AI's representation of the researcher. Extensions of the concept include communication between living researchers and their electronic copies, collaboration among individual electronic copies, as well as the creation and interconnection of organizational electronic copies to optimize information access and strategic decision-making. Ethical considerations such as ownership and security of these electronic copies are highlighted as critical for responsible implementation. The findings suggest promising opportunities for AI-driven preservation and augmentation of intellectual legacy. 

**Abstract (ZH)**: 本研究探讨了通过训练人工智能（AI）在已故研究人员个人计算机中存储的数据上创建“电子副本”的可行性。通过分析继承的研究人员计算机上的典型数据量，包括文章、电子邮件和草稿等文本文件，估计可用于AI训练的数据量约为一百万字。这一数据量足以对如GPT-4等先进预训练模型进行微调，从而使AI能够高保真地复制研究人员的写作风格、专业领域知识和修辞voice。研究还讨论了包括非文本数据和文件元数据在内的潜在增强功能，以丰富对研究人员的AI表示。该概念的拓展包括生者研究人员与其电子副本之间的交流、个体电子副本之间的协作，以及组织电子副本的创建和互联，以优化信息访问和战略决策。研究还强调了产权和安全等伦理考虑对于负责任实施的重要性，并指出AI驱动的知识保存与增益具有广阔前景。 

---
# Automated Grading of Students' Handwritten Graphs: A Comparison of Meta-Learning and Vision-Large Language Models 

**Title (ZH)**: 基于元学习和视觉大语言模型的学生手写图表的自动评分比较 

**Authors**: Behnam Parsaeifard, Martin Hlosta, Per Bergamin  

**Link**: [PDF](https://arxiv.org/pdf/2507.03056)  

**Abstract**: With the rise of online learning, the demand for efficient and consistent assessment in mathematics has significantly increased over the past decade. Machine Learning (ML), particularly Natural Language Processing (NLP), has been widely used for autograding student responses, particularly those involving text and/or mathematical expressions. However, there has been limited research on autograding responses involving students' handwritten graphs, despite their prevalence in Science, Technology, Engineering, and Mathematics (STEM) curricula. In this study, we implement multimodal meta-learning models for autograding images containing students' handwritten graphs and text. We further compare the performance of Vision Large Language Models (VLLMs) with these specially trained metalearning models. Our results, evaluated on a real-world dataset collected from our institution, show that the best-performing meta-learning models outperform VLLMs in 2-way classification tasks. In contrast, in more complex 3-way classification tasks, the best-performing VLLMs slightly outperform the meta-learning models. While VLLMs show promising results, their reliability and practical applicability remain uncertain and require further investigation. 

**Abstract (ZH)**: 随着在线学习的兴起，过去十年中对数学高效一致评估的需求显著增加。机器学习（ML），特别是自然语言处理（NLP），已被广泛用于自动批改学生的答案，尤其是涉及文本和/或数学表达式的情况。然而，对于涉及学生手绘图表的答案批改，尽管这类图表在STEM课程中普遍存在，关于这方面的研究仍相对有限。在本研究中，我们实现了多模态元学习模型来自动批改包含学生手绘图表和文本的图像。我们进一步将视觉大型语言模型（VLLMs）的性能与其特别训练的元学习模型进行了比较。我们的结果，在我们机构收集的真实数据集上评估，显示最佳的元学习模型在二分类任务中优于VLLMs；而在更复杂的三分类任务中，最佳的VLLMs略微优于元学习模型。尽管VLLMs展现出良好的前景，但它们的可靠性和实际应用性仍有待进一步调查。 

---
# LATTE: Latent Trajectory Embedding for Diffusion-Generated Image Detection 

**Title (ZH)**: LATTE: 潜在轨迹嵌入用于扩散生成图像检测 

**Authors**: Ana Vasilcoiu, Ivona Najdenkoska, Zeno Geradts, Marcel Worring  

**Link**: [PDF](https://arxiv.org/pdf/2507.03054)  

**Abstract**: The rapid advancement of diffusion-based image generators has made it increasingly difficult to distinguish generated from real images. This can erode trust in digital media, making it critical to develop generalizable detectors for generated images. Recent methods leverage diffusion denoising cues, but mainly focus on single-step reconstruction errors, ignoring the inherent sequential nature of the denoising process. In this work, we propose LATTE - Latent Trajectory Embedding - a novel approach that models the evolution of latent embeddings across several denoising timesteps. By modeling the trajectory of such embeddings rather than single-step errors, LATTE captures subtle, discriminative patterns that distinguish real from generated images. Each latent is refined by employing our latent-visual feature refinement module and aggregated into a unified representation. Afterwards, it is fused with the visual features and finally passed into a lightweight classifier. Our experiments demonstrate that LATTE surpasses the baselines on several established benchmarks, such as GenImage and DiffusionFake. Moreover, it demonstrates strong performance in cross-generator and cross-datasets settings, highlighting the potential of using the trajectory of latent embeddings for generated image detection. The code is available on the following link: this https URL. 

**Abstract (ZH)**: 基于扩散的图像生成器的快速进步使区分生成图像和真实图像变得越来越困难，这会侵蚀数字媒体的信任，因此开发可用于生成图像检测的一般化检测器变得至关重要。近期方法利用去噪提示，但主要集中在单步重建误差，忽视了去噪过程的固有顺序性。本文提出LATTE - 潜在轨迹嵌入 - 一种新颖的方法，用于建模多次去噪时间步长中潜在嵌入物的演变。通过建模嵌入物的轨迹而非单步误差，LATTE 捕获区分真实图像和生成图像的微妙且具有区别的模式。每个潜在嵌入物通过我们提出的空间潜在特征精炼模块进行精炼并聚合为统一表示。之后，将其与视觉特征融合，最终传入轻量级分类器。实验结果表明，LATTE 在多个基准测试（如 GenImage 和 DiffusionFake）上优于基准方法，同时在跨生成器和跨数据集场景中表现出强大的性能，突显了使用潜在嵌入物的轨迹进行生成图像检测的潜力。代码可通过以下链接访问：this https URL。 

---
# From 2:4 to 8:16 sparsity patterns in LLMs for Outliers and Weights with Variance Correction 

**Title (ZH)**: 从2:4到8:16稀疏模式在考虑方差修正的情况下应用于LLMs的异常值和权重 

**Authors**: Egor Maximov, Yulia Kuzkina, Azamat Kanametov, Alexander Prutko, Aleksei Goncharov, Maxim Zhelnin, Egor Shvetsov  

**Link**: [PDF](https://arxiv.org/pdf/2507.03052)  

**Abstract**: As large language models (LLMs) grow in size, efficient compression techniques like quantization and sparsification are critical. While quantization maintains performance with reduced precision, structured sparsity methods, such as N:M sparsification, often fall short due to limited flexibility, and sensitivity to outlier weights. We explore 8:16 semi-structured sparsity, demonstrating its ability to surpass the Performance Threshold-where a compressed model matches the accuracy of its uncompressed or smaller counterpart under equivalent memory constraints. Compared to 2:4 sparsity, 8:16 offers greater flexibility with minimal storage overhead (0.875 vs. 0.75 bits/element). We also apply sparse structured patterns for salient weights, showing that structured sparsity for outliers is competitive with unstructured approaches leading to equivalent or better results. Finally, we demonstrate that simple techniques such as variance correction and SmoothQuant like weight equalization improve sparse models performance. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）规模扩大，高效压缩技术如量化和稀疏化至关重要。尽管量化可以在降低精度的同时保持性能，但结构化稀疏化方法，如N:M稀疏化，往往因为灵活性有限和对异常权重敏感而效果不佳。我们探索了8:16半结构化稀疏化，证明了其能够在等内存约束条件下超越性能阈值，即压缩模型的准确度与其未压缩或更小的版本相当。与2:4稀疏化相比，8:16提供了更大的灵活性，并且存储开销较小（0.875 vs. 0.75位/元素）。我们还应用了突出权重的结构化稀疏模式，表明针对异常权重的结构化稀疏化与非结构化方法具有竞争力，导致等同或更好的结果。最后，我们展示了诸如方差校正和类似于权重均衡的简单技术可以提高稀疏模型的性能。 

---
# Improving LLM Reasoning for Vulnerability Detection via Group Relative Policy Optimization 

**Title (ZH)**: 基于组相对策略优化提高LLM推理以增强漏洞检测 

**Authors**: Marco Simoni, Aleksandar Fontana, Giulio Rossolini, Andrea Saracino  

**Link**: [PDF](https://arxiv.org/pdf/2507.03051)  

**Abstract**: Improving and understanding the training dynamics and reasoning of Large Language Models (LLMs) has become essential for their deployment in AI-based security tools, such as software vulnerability detection. In this work, we present an extensive study aimed at advancing recent RL-based finetuning techniques for LLMs in the context of vulnerability detection.
We start by highlighting key limitations of commonly adopted LLMs, such as their tendency to over-predict certain types of vulnerabilities while failing to detect others. To address this challenge, we explore the use of Group Relative Policy Optimization (GRPO), a recent policy-gradient method, for guiding LLM behavior through structured, rule-based rewards. We enable its application to the vulnerability detection task by redefining its advantage functions and reward signals using annotations from widely used datasets in the field, including BigVul, DiverseVul, and CleanVul.
The proposed methodology enables an extensive set of experiments, addressing multiple research questions regarding the impact of GRPO on generalization, reasoning capabilities, and performance improvements over standard supervised finetuning (SFT). Our findings offer valuable insights into the potential of RL-based training to enhance both the performance and reasoning abilities of LLMs in the context of software vulnerability detection. 

**Abstract (ZH)**: 改善和理解大型语言模型（LLMs）的训练动态和推理能力对于它们在基于AI的安全工具中的部署变得至关重要，尤其是在软件漏洞检测方面。在本工作中，我们提出了一个全面的研究，旨在推进针对漏洞检测的基于强化学习（RL）的LLM微调技术。

已突显了广泛采用的LLMs的关键局限性，如其倾向于高估某些类型的漏洞，而未能检测其他类型的漏洞。为解决这一挑战，我们探索了使用组相对策略优化（GRPO），一种近期的策略梯度方法，通过结构化的规则基础奖励来引导LLM的行为。我们通过使用来自该领域广泛使用的数据集（包括BigVul、DiverseVul和CleanVul）的注释来重新定义其优势函数和奖励信号，使其应用于漏洞检测任务。

所提出的方法使得进行大量实验成为可能，这些问题涉及GRPO对泛化、推理能力和相对于标准监督微调（SFT）的性能改进的影响。我们的研究结果提供了关于基于RL的训练如何在软件漏洞检测上下文中增强LLM的性能和推理能力的重要见解。 

---
# From Turing to Tomorrow: The UK's Approach to AI Regulation 

**Title (ZH)**: 从图灵到未来：英国的AI监管之道 

**Authors**: Oliver Ritchie, Markus Anderljung, Tom Rachman  

**Link**: [PDF](https://arxiv.org/pdf/2507.03050)  

**Abstract**: The UK has pursued a distinctive path in AI regulation: less cautious than the EU but more willing to address risks than the US, and has emerged as a global leader in coordinating AI safety efforts. Impressive developments from companies like London-based DeepMind began to spark concerns in the UK about catastrophic risks from around 2012, although regulatory discussion at the time focussed on bias and discrimination. By 2022, these discussions had evolved into a "pro-innovation" strategy, in which the government directed existing regulators to take a light-touch approach, governing AI at point of use, but avoided regulating the technology or infrastructure directly. ChatGPT arrived in late 2022, galvanising concerns that this approach may be insufficient. The UK responded by establishing an AI Safety Institute to monitor risks and hosting the first international AI Safety Summit in 2023, but - unlike the EU - refrained from regulating frontier AI development in addition to its use. A new government was elected in 2024 which promised to address this gap, but at the time of writing is yet to do so.
What should the UK do next? The government faces competing objectives: harnessing AI for economic growth and better public services while mitigating risk. In light of these, we propose establishing a flexible, principles-based regulator to oversee the most advanced AI development, defensive measures against risks from AI-enabled biological design tools, and argue that more technical work is needed to understand how to respond to AI-generated misinformation. We argue for updated legal frameworks on copyright, discrimination, and AI agents, and that regulators will have a limited but important role if AI substantially disrupts labour markets.
If the UK gets AI regulation right, it could demonstrate how democratic societies can harness AI's benefits while managing its risks. 

**Abstract (ZH)**: 英国在AI监管方面遵循了一条独特的路径：比欧盟更为大胆，但比美国更愿意应对风险，并已成为协调AI安全努力的全球领导者。从2012年起，以伦敦DeepMind为代表公司的显著进展开始引发英国关于灾难性风险的担忧，尽管当时的监管讨论主要集中在偏见和歧视问题上。到2022年，这些讨论已经转变为一种“促进创新”的策略，在这种策略中，政府指导现有的监管机构采取以用户使用为导向的轻触监管方式，但避免直接监管技术或基础设施。ChatGPT于2022年末推出，引发了对这种做法是否足够的担忧。英国随后成立了AI安全研究所以监测风险，并于2023年举办了第一届国际AI安全峰会，但不像欧盟，英国没有对AI前沿开发进行额外的监管。2024年新政府上台承诺解决这一差距，但截至撰文时尚未采取行动。
英国下一步应该怎么做？政府面临着相互竞争的目标：利用AI促进经济增长和提高公共服务水平，同时减少风险。鉴于这些目标，我们建议建立一个灵活的原则性监管机构来监督最先进的人工智能开发，以及针对人工智能驱动的生物设计工具带来的风险采取防御性措施，并认为需要更多技术工作来理解如何应对由人工智能生成的虚假信息。我们主张更新版权法、反歧视法和人工智能代理法，并认为在人工智能显著扰乱劳动力市场的情况下，监管机构将扮演有限但重要的角色。
如果英国在AI监管方面取得成功，它将展示民主社会如何利用AI的好处同时管理其风险。 

---
# Personalised Explanations in Long-term Human-Robot Interactions 

**Title (ZH)**: 长期内个性化解释的人机互动 

**Authors**: Ferran Gebellí, Anaís Garrell, Jan-Gerrit Habekost, Séverin Lemaignan, Stefan Wermter, Raquel Ros  

**Link**: [PDF](https://arxiv.org/pdf/2507.03049)  

**Abstract**: In the field of Human-Robot Interaction (HRI), a fundamental challenge is to facilitate human understanding of robots. The emerging domain of eXplainable HRI (XHRI) investigates methods to generate explanations and evaluate their impact on human-robot interactions. Previous works have highlighted the need to personalise the level of detail of these explanations to enhance usability and comprehension. Our paper presents a framework designed to update and retrieve user knowledge-memory models, allowing for adapting the explanations' level of detail while referencing previously acquired concepts. Three architectures based on our proposed framework that use Large Language Models (LLMs) are evaluated in two distinct scenarios: a hospital patrolling robot and a kitchen assistant robot. Experimental results demonstrate that a two-stage architecture, which first generates an explanation and then personalises it, is the framework architecture that effectively reduces the level of detail only when there is related user knowledge. 

**Abstract (ZH)**: 在人机交互（HRI）领域，一个基本挑战是促进人类对机器人的理解。可解释人机交互（XHRI）这一新兴领域研究生成解释的方法及其对人机交互影响的评估。先前的研究强调需要个性化这些解释的详细程度以提高可用性和理解度。本文提出了一种框架，用于更新和检索用户知识-记忆模型，使解释的详细程度能够根据不同用户的先前概念进行调整。基于本文提出框架的三种使用大型语言模型（LLMs）的架构在两种不同场景中进行了评估：巡逻机器人和厨房助手机器人。实验结果表明，两阶段架构，首先生成解释然后个性化解释，是在用户具有相关知识时有效降低解释详细程度的框架架构。 

---
# Monitoring of Static Fairness 

**Title (ZH)**: 静态公平性监测 

**Authors**: Thomas A. Henzinger, Mahyar Karimi, Konstantin Kueffner, Kaushik Mallik  

**Link**: [PDF](https://arxiv.org/pdf/2507.03048)  

**Abstract**: Machine-learned systems are in widespread use for making decisions about humans, and it is important that they are fair, i.e., not biased against individuals based on sensitive attributes.
We present a general framework of runtime verification of algorithmic fairness for systems whose models are unknown, but are assumed to have a Markov chain structure, with or without full observation of the state space.
We introduce a specification language that can model many common algorithmic fairness properties, such as demographic parity, equal opportunity, and social burden.
We build monitors that observe a long sequence of events as generated by a given system, and output, after each observation, a quantitative estimate of how fair or biased the system was on that run until that point in time.
The estimate is proven to be correct modulo a variable error bound and a given confidence level, where the error bound gets tighter as the observed sequence gets longer.
We present two categories of monitoring algorithms, namely ones with a uniform error bound across all time points, and ones with weaker non-uniform, pointwise error bounds at different time points.
Our monitoring algorithms use statistical tools that are adapted to suit the dynamic requirements of monitoring and the special needs of the fairness specifications.
Using a prototype implementation, we show how we can monitor if a bank is fair in giving loans to applicants from different social backgrounds, and if a college is fair in admitting students while maintaining a reasonable financial burden on the society.
In these experiments, our monitors took less than a millisecond to update their verdicts after each observation. 

**Abstract (ZH)**: 基于马尔可夫链结构的未知模型系统运行时算法公平性验证框架 

---
# Counterfactual Tuning for Temporal Sensitivity Enhancement in Large Language Model-based Recommendation 

**Title (ZH)**: 基于大型语言模型的推荐中时空敏感性增强的反事实调优 

**Authors**: Yutian Liu, Zhengyi Yang, Jiancan Wu, Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.03047)  

**Abstract**: Recent advances have applied large language models (LLMs) to sequential recommendation, leveraging their pre-training knowledge and reasoning capabilities to provide more personalized user experiences. However, existing LLM-based methods fail to sufficiently leverage the rich temporal information inherent in users' historical interaction sequences, stemming from fundamental architectural constraints: LLMs process information through self-attention mechanisms that lack inherent sequence ordering and rely on position embeddings designed primarily for natural language rather than user interaction sequences. This limitation significantly impairs their ability to capture the evolution of user preferences over time and predict future interests accurately.
To address this critical gap, we propose Counterfactual Enhanced Temporal Framework for LLM-Based Recommendation (CETRec). CETRec is grounded in causal inference principles, which allow it to isolate and measure the specific impact of temporal information on recommendation outcomes. By conceptualizing temporal order as an independent causal factor distinct from item content, we can quantify its unique contribution through counterfactual reasoning--comparing what recommendations would be made with and without temporal information while keeping all other factors constant. This causal framing enables CETRec to design a novel counterfactual tuning objective that directly optimizes the model's temporal sensitivity, teaching LLMs to recognize both absolute timestamps and relative ordering patterns in user histories. Combined with our counterfactual tuning task derived from causal analysis, CETRec effectively enhances LLMs' awareness of both absolute order (how recently items were interacted with) and relative order (the sequential relationships between items). 

**Abstract (ZH)**: 基于大型语言模型的计因增强时序推荐框架（CETRec） 

---
# Optimisation Is Not What You Need 

**Title (ZH)**: 优化并不是你需要的 

**Authors**: Alfredo Ibias  

**Link**: [PDF](https://arxiv.org/pdf/2507.03045)  

**Abstract**: The Artificial Intelligence field has focused on developing optimisation methods to solve multiple problems, specifically problems that we thought to be only solvable through cognition. The obtained results have been outstanding, being able to even surpass the Turing Test. However, we have found that these optimisation methods share some fundamental flaws that impede them to become a true artificial cognition. Specifically, the field have identified catastrophic forgetting as a fundamental problem to develop such cognition. This paper formally proves that this problem is inherent to optimisation methods, and as such it will always limit approaches that try to solve the Artificial General Intelligence problem as an optimisation problem. Additionally, it addresses the problem of overfitting and discuss about other smaller problems that optimisation methods pose. Finally, it empirically shows how world-modelling methods avoid suffering from either problem. As a conclusion, the field of Artificial Intelligence needs to look outside the machine learning field to find methods capable of developing an artificial cognition. 

**Abstract (ZH)**: 人工智能领域专注于开发优化方法以解决多种问题，特别是在我们认为只有通过认知才能解决的问题上。取得的结果非常出色，甚至能够超越图灵测试。然而，我们发现这些优化方法存在一些基本缺陷，阻碍它们成为真正的 artificial cognition。具体地说，该领域已识别出灾难性遗忘是发展这种认知的根本问题。本文正式证明了这个问题是优化方法的固有缺陷，因此它将始终限制那些试图将通用人工智能问题作为优化问题来解决的方法。此外，本文还探讨了优化方法存在的过拟合问题以及其他较小的问题，并实证展示了世界建模方法如何避免遭受这些问题。总之，人工智能领域需要在机器学习领域之外寻找能够发展 artificial cognition 的方法。 

---
# K-Function: Joint Pronunciation Transcription and Feedback for Evaluating Kids Language Function 

**Title (ZH)**: K-函数：联合发音转录与反馈以评估儿童语言功能 

**Authors**: Shuhe Li, Chenxu Guo, Jiachen Lian, Cheol Jun Cho, Wenshuo Zhao, Xuanru Zhou, Dingkun Zhou, Sam Wang, Grace Wang, Jingze Yang, Jingyi Xu, Ruohan Bao, Elise Brenner, Brandon In, Francesca Pei, Maria Luisa Gorno-Tempini, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2507.03043)  

**Abstract**: Early evaluation of children's language is frustrated by the high pitch, long phones, and sparse data that derail automatic speech recognisers. We introduce K-Function, a unified framework that combines accurate sub-word transcription, objective scoring, and actionable feedback. Its core, Kids-WFST, merges a Wav2Vec2 phoneme encoder with a phoneme-similarity Dysfluent-WFST to capture child-specific errors while remaining fully interpretable. Kids-WFST attains 1.39% phoneme error on MyST and 8.61% on Multitudes--absolute gains of 10.47 and 7.06 points over a greedy-search decoder. These high-fidelity transcripts power an LLM that grades verbal skills, milestones, reading, and comprehension, aligning with human proctors and supplying tongue-and-lip visualizations plus targeted advice. The results show that precise phoneme recognition cements a complete diagnostic-feedback loop, paving the way for scalable, clinician-ready language assessment. 

**Abstract (ZH)**: 早评价儿童语言能力受制于高音调、长音素和稀疏数据对自动语音识别系统的干扰。我们提出了K-Function，这是一个结合了准确的子词转录、客观评分和可操作反馈的统一框架。其核心组件Kids-WFST将Wav2Vec2音素编码器与音素相似性失言WFST相结合，以捕捉儿童特有的错误，同时保持完全可解释性。Kids-WFST在MyST上的音素错误率为1.39%，在Multitudes上的音素错误率为8.61%，分别比贪婪搜索解码器提高了10.47和7.06个百分点。这些高保真转录文本驱动了一个大型语言模型，用于评估口头技能、里程碑、阅读能力和理解能力，并与人类考官对齐，提供舌唇可视化及针对性建议。结果表明，精确的音素识别确立了完整的诊断反馈循环，为可扩展的、面向临床的语言评估铺平了道路。 

---
# Dynamic Long Short-Term Memory Based Memory Storage For Long Horizon LLM Interaction 

**Title (ZH)**: 基于动态长短期记忆的内存存储方案以支持长远 horizon LLM 交互 

**Authors**: Yuyang Lou, Charles Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.03042)  

**Abstract**: Memory storage for Large Language models (LLMs) is becoming an increasingly active area of research, particularly for enabling personalization across long conversations. We propose Pref-LSTM, a dynamic and lightweight framework that combines a BERT-based classifier with a LSTM memory module that generates memory embedding which then is soft-prompt injected into a frozen LLM. We synthetically curate a dataset of preference and non-preference conversation turns to train our BERT-based classifier. Although our LSTM-based memory encoder did not yield strong results, we find that the BERT-based classifier performs reliably in identifying explicit and implicit user preferences. Our research demonstrates the viability of using preference filtering with LSTM gating principals as an efficient path towards scalable user preference modeling, without extensive overhead and fine-tuning. 

**Abstract (ZH)**: 大型语言模型（LLMs）的内存存储成为一项日益活跃的研究领域，特别是在长对话中实现个性化方面。我们提出了一种名为Pref-LSTM的动态轻量级框架，该框架结合了基于BERT的分类器和一个基于LSTM的记忆模块，该模块生成记忆嵌入，然后将其软提示注入冻结的LLM中。我们合成了一组偏好和非偏好对话片段数据集来训练我们的基于BERT的分类器。尽管我们的基于LSTM的记忆编码器没有取得显著结果，但我们发现基于BERT的分类器在识别显性和隐性用户偏好方面表现可靠。我们的研究证明了使用偏好过滤和LSTM门控原理进行可扩展用户偏好建模的有效途径，无需大量额外开销和微调。 

---
# Optimas: Optimizing Compound AI Systems with Globally Aligned Local Rewards 

**Title (ZH)**: Optimas：以全局对齐的局部奖励优化复合AI系统 

**Authors**: Shirley Wu, Parth Sarthi, Shiyu Zhao, Aaron Lee, Herumb Shandilya, Adrian Mladenic Grobelnik, Nurendra Choudhary, Eddie Huang, Karthik Subbian, Linjun Zhang, Diyi Yang, James Zou, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2507.03041)  

**Abstract**: Compound AI systems integrating multiple components, such as Large Language Models, specialized tools, and traditional machine learning models, are increasingly deployed to solve complex real-world tasks. However, optimizing compound systems remains challenging due to their non-differentiable structures and diverse configuration types across components, including prompts, hyperparameters, and model parameters. To address this challenge, we propose Optimas, a unified framework for effective optimization of compound systems. The core idea of Optimas is to maintain one Local Reward Function (LRF) per component, each satisfying a local-global alignment property, i.e., each component's local reward correlates with the global system performance. In each iteration, Optimas efficiently adapts the LRFs to maintain this property while simultaneously maximizing each component's local reward. This approach enables independent updates of heterogeneous configurations using the designated optimization method, while ensuring that local improvements consistently lead to performance gains. We present extensive evaluations across five real-world compound systems to demonstrate that Optimas outperforms strong baselines by an average improvement of 11.92%, offering a general and effective approach for improving compound systems. Our website is at this https URL. 

**Abstract (ZH)**: 综合多个组件（如大型语言模型、专业工具和传统机器学习模型）的AI系统日益用于解决复杂的现实任务。然而，由于这些系统具有非可微结构并且各个组件（包括提示、超参数和模型参数）的配置类型多样，因此优化这些系统仍然是一个挑战。为了应对这一挑战，我们提出了一种名为Optimas的统一框架，用于有效优化复合系统。Optimas的核心思想是为每个组件维护一个局部奖励函数（LRF），每个LRF都满足局部-全局对齐属性，即每个组件的局部奖励与系统整体性能相关。在每次迭代中，Optimas高效地适应LRFs以保持这一属性，同时最大限度地提高每个组件的局部奖励。这种方法允许使用指定的优化方法独立更新异构配置，并确保局部改进始终带来性能提升。我们在五个实际的复合系统上进行了广泛的评估，结果表明，Optimas平均优于强基线11.92%，提供了一种通用且有效的方法来提升复合系统。我们的网站地址为：这个https URL。 

---
