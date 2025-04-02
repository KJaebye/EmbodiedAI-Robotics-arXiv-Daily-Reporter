# AI Judges in Design: Statistical Perspectives on Achieving Human Expert Equivalence With Vision-Language Models 

**Title (ZH)**: AI法官在设计中的角色：视觉-语言模型在实现人类专家等效性方面的统计视角 

**Authors**: Kristen M. Edwards, Farnaz Tehranchi, Scarlett R. Miller, Faez Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2504.00938)  

**Abstract**: The subjective evaluation of early stage engineering designs, such as conceptual sketches, traditionally relies on human experts. However, expert evaluations are time-consuming, expensive, and sometimes inconsistent. Recent advances in vision-language models (VLMs) offer the potential to automate design assessments, but it is crucial to ensure that these AI ``judges'' perform on par with human experts. However, no existing framework assesses expert equivalence. This paper introduces a rigorous statistical framework to determine whether an AI judge's ratings match those of human experts. We apply this framework in a case study evaluating four VLM-based judges on key design metrics (uniqueness, creativity, usefulness, and drawing quality). These AI judges employ various in-context learning (ICL) techniques, including uni- vs. multimodal prompts and inference-time reasoning. The same statistical framework is used to assess three trained novices for expert-equivalence. Results show that the top-performing AI judge, using text- and image-based ICL with reasoning, achieves expert-level agreement for uniqueness and drawing quality and outperforms or matches trained novices across all metrics. In 6/6 runs for both uniqueness and creativity, and 5/6 runs for both drawing quality and usefulness, its agreement with experts meets or exceeds that of the majority of trained novices. These findings suggest that reasoning-supported VLM models can achieve human-expert equivalence in design evaluation. This has implications for scaling design evaluation in education and practice, and provides a general statistical framework for validating AI judges in other domains requiring subjective content evaluation. 

**Abstract (ZH)**: 基于视觉-语言模型的早期工程设计主观评价：评估AI评判员与人类专家的等效性 

---
# Grounding Multimodal LLMs to Embodied Agents that Ask for Help with Reinforcement Learning 

**Title (ZH)**: 将多模态LLMgrounding到寻求帮助的具身代理上，并应用于强化学习 

**Authors**: Ram Ramrakhya, Matthew Chang, Xavier Puig, Ruta Desai, Zsolt Kira, Roozbeh Mottaghi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00907)  

**Abstract**: Embodied agents operating in real-world environments must interpret ambiguous and under-specified human instructions. A capable household robot should recognize ambiguity and ask relevant clarification questions to infer the user intent accurately, leading to more effective task execution. To study this problem, we introduce the Ask-to-Act task, where an embodied agent must fetch a specific object instance given an ambiguous instruction in a home environment. The agent must strategically ask minimal, yet relevant, clarification questions to resolve ambiguity while navigating under partial observability. To solve this problem, we propose a novel approach that fine-tunes multimodal large language models (MLLMs) as vision-language-action (VLA) policies using online reinforcement learning (RL) with LLM-generated rewards. Our method eliminates the need for large-scale human demonstrations or manually engineered rewards for training such agents. We benchmark against strong zero-shot baselines, including GPT-4o, and supervised fine-tuned MLLMs, on our task. Our results demonstrate that our RL-finetuned MLLM outperforms all baselines by a significant margin ($19.1$-$40.3\%$), generalizing well to novel scenes and tasks. To the best of our knowledge, this is the first demonstration of adapting MLLMs as VLA agents that can act and ask for help using LLM-generated rewards with online RL. 

**Abstract (ZH)**: 具身代理在现实世界环境中操作时必须解释模糊和不明确的人类指令。一个 capable 的家用机器人应该识别出指令的模糊性，并提出相关的问题以准确推断用户意图，从而提高任务执行的有效性。为研究这一问题，我们引入了“Act and Ask”任务，即在一个家庭环境中，具身代理必须根据模糊的指令获取特定物体实例。代理必须在部分可观测性下，战略性地提出最少但相关的问题以解决模糊性。为解决这一问题，我们提出了一种新的方法，利用大型语言模型（LLM）的在线强化学习（RL）微调，将其作为视觉-语言-行动（VLA）策略使用，同时使用LLM生成的奖励。我们的方法消除了大规模人工演示或手动工程奖励的需要。我们在我们的任务上与强零-shot 基线（包括GPT-4o）和监督微调的LLM进行评估。结果表明，我们的RL微调LLM在所有基线中表现出显著的领先优势（19.1%-40.3%），并能很好地泛化到新的场景和任务中。据我们所知，这是首次将LLM适应为能够使用LLM生成奖励进行在线RL的VLA代理并能主动寻求帮助的示范。 

---
# Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents 

**Title (ZH)**: Agent S2: 一种计算机使用代理的组合通用专家框架 

**Authors**: Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang, Ang Li, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00906)  

**Abstract**: Computer use agents automate digital tasks by directly interacting with graphical user interfaces (GUIs) on computers and mobile devices, offering significant potential to enhance human productivity by completing an open-ended space of user queries. However, current agents face significant challenges: imprecise grounding of GUI elements, difficulties with long-horizon task planning, and performance bottlenecks from relying on single generalist models for diverse cognitive tasks. To this end, we introduce Agent S2, a novel compositional framework that delegates cognitive responsibilities across various generalist and specialist models. We propose a novel Mixture-of-Grounding technique to achieve precise GUI localization and introduce Proactive Hierarchical Planning, dynamically refining action plans at multiple temporal scales in response to evolving observations. Evaluations demonstrate that Agent S2 establishes new state-of-the-art (SOTA) performance on three prominent computer use benchmarks. Specifically, Agent S2 achieves 18.9% and 32.7% relative improvements over leading baseline agents such as Claude Computer Use and UI-TARS on the OSWorld 15-step and 50-step evaluation. Moreover, Agent S2 generalizes effectively to other operating systems and applications, surpassing previous best methods by 52.8% on WindowsAgentArena and by 16.52% on AndroidWorld relatively. Code available at this https URL. 

**Abstract (ZH)**: Agent S2：一种新型的 compositional 框架，通过分配认知责任来增强数字任务自动化 

---
# Investigating Large Language Models in Diagnosing Students' Cognitive Skills in Math Problem-solving 

**Title (ZH)**: 探究大型语言模型在数学问题解决中诊断学生认知技能的应用 

**Authors**: Hyoungwook Jin, Yoonsu Kim, Dongyun Jung, Seungju Kim, Kiyoon Choi, Jinho Son, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.00843)  

**Abstract**: Mathematics learning entails mastery of both content knowledge and cognitive processing of knowing, applying, and reasoning with it. Automated math assessment primarily has focused on grading students' exhibition of content knowledge by finding textual evidence, such as specific numbers, formulas, and statements. Recent advancements in problem-solving, image recognition, and reasoning capabilities of large language models (LLMs) show promise for nuanced evaluation of students' cognitive skills. Diagnosing cognitive skills needs to infer students' thinking processes beyond textual evidence, which is an underexplored task in LLM-based automated assessment. In this work, we investigate how state-of-the-art LLMs diagnose students' cognitive skills in mathematics. We constructed MathCog, a novel benchmark dataset comprising 639 student responses to 110 expert-curated middle school math problems, each annotated with detailed teachers' diagnoses based on cognitive skill checklists. Using MathCog, we evaluated 16 closed and open LLMs of varying model sizes and vendors. Our evaluation reveals that even the state-of-the-art LLMs struggle with the task, all F1 scores below 0.5, and tend to exhibit strong false confidence for incorrect cases ($r_s=.617$). We also found that model size positively correlates with the diagnosis performance ($r_s=.771$). Finally, we discuss the implications of these findings, the overconfidence issue, and directions for improving automated cognitive skill diagnosis. 

**Abstract (ZH)**: 数学学习涉及内容知识的掌握和认知处理能力的培养，包括应用和推理。自动数学评估主要侧重于通过查找特定数字、公式和陈述等文本证据来评估学生对内容知识的展示。大型语言模型（LLMs）在解决问题、图像识别和推理能力方面的最新进展显示出对学生的认知技能进行细致评估的潜力。诊断认知技能需要推断学生的思想过程，这在基于LLM的自动评估中是一个未被充分探索的任务。在本文中，我们研究了最先进的LLMs如何诊断学生在数学中的认知技能。我们构建了MathCog，这是一个新颖的基准数据集，包含639名学生的110个中学数学问题的响应，每个问题都基于认知技能清单进行了详细的教师诊断标注。使用MathCog，我们评估了16个不同模型大小和供应商的封闭和开放LLMs。评估结果显示，即使是最先进的LLMs也难以完成这项任务，所有F1分数均低于0.5，并且在错误情况下表现出强烈的事后自信（$r_s=.617$）。我们还发现，模型大小与诊断性能正相关（$r_s=.771$）。最后，我们讨论了这些发现的意义、过自信问题以及改进自动认知技能诊断的方向。 

---
# Example-Based Concept Analysis Framework for Deep Weather Forecast Models 

**Title (ZH)**: 基于示例的概念分析框架：深水气象预报模型 

**Authors**: Soyeon Kim, Junho Choi, Subeen Lee, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00831)  

**Abstract**: To improve the trustworthiness of an AI model, finding consistent, understandable representations of its inference process is essential. This understanding is particularly important in high-stakes operations such as weather forecasting, where the identification of underlying meteorological mechanisms is as critical as the accuracy of the predictions. Despite the growing literature that addresses this issue through explainable AI, the applicability of their solutions is often limited due to their AI-centric development. To fill this gap, we follow a user-centric process to develop an example-based concept analysis framework, which identifies cases that follow a similar inference process as the target instance in a target model and presents them in a user-comprehensible format. Our framework provides the users with visually and conceptually analogous examples, including the probability of concept assignment to resolve ambiguities in weather mechanisms. To bridge the gap between vector representations identified from models and human-understandable explanations, we compile a human-annotated concept dataset and implement a user interface to assist domain experts involved in the the framework development. 

**Abstract (ZH)**: 提高AI模型可信度的关键在于找到其推理过程的一致且可理解的表示形式。这一理解在如天气预报等高风险操作中尤为重要，因为识别潜在的气象机制与预测的准确性一样重要。尽管已有大量关于此问题的可解释AI研究，但其解决方案的应用受限，因为这些解决方案多以AI为中心进行开发。为弥补这一差距，我们采用用户为中心的过程，开发了一种基于实例的概念分析框架。该框架识别出与目标模型中目标实例具有相似推理过程的案例，并以用户可理解的格式呈现。我们的框架提供了可视化和概念上类比的示例，包括概念的赋值概率，以解决天气机制中的模糊性。为了弥合来自模型的向量表示与人类可理解解释之间的差距，我们构建了一个由人工标注的概念数据集，并实现了一个用户界面，以辅助参与框架开发的领域专家。 

---
# Explainable AI-Based Interface System for Weather Forecasting Model 

**Title (ZH)**: 基于可解释AI的天气预报模型界面系统 

**Authors**: Soyeon Kim, Junho Choi, Yeji Choi, Subeen Lee, Artyom Stitsyuk, Minkyoung Park, Seongyeop Jeong, Youhyun Baek, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00795)  

**Abstract**: Machine learning (ML) is becoming increasingly popular in meteorological decision-making. Although the literature on explainable artificial intelligence (XAI) is growing steadily, user-centered XAI studies have not extend to this domain yet. This study defines three requirements for explanations of black-box models in meteorology through user studies: statistical model performance for different rainfall scenarios to identify model bias, model reasoning, and the confidence of model outputs. Appropriate XAI methods are mapped to each requirement, and the generated explanations are tested quantitatively and qualitatively. An XAI interface system is designed based on user feedback. The results indicate that the explanations increase decision utility and user trust. Users prefer intuitive explanations over those based on XAI algorithms even for potentially easy-to-recognize examples. These findings can provide evidence for future research on user-centered XAI algorithms, as well as a basis to improve the usability of AI systems in practice. 

**Abstract (ZH)**: 机器学习在气象决策中的应用日益增多。尽管可解释人工智能（XAI）的相关文献在稳步增长，但用户中心的XAI研究尚未扩展到该领域。本研究通过用户研究定义了气象中黑盒模型解释的三项要求：不同降雨情景下的统计模型性能以识别模型偏差、模型推理以及模型输出的信心。合适的方法被映射到每个要求，并生成的解释进行了定量和定性的测试。基于用户反馈设计了XAI界面系统。结果显示，这些解释增加了决策的价值和用户的信任。用户更偏好直观的解释，即使对于可能容易识别的例子也是如此。这些发现可以为未来用户中心的XAI算法研究提供证据，并为改进实践中人工智能系统的可用性提供基础。 

---
# Do We Truly Need So Many Samples? Multi-LLM Repeated Sampling Efficiently Scale Test-Time Compute 

**Title (ZH)**: 我们需要这么多样本吗？多模态大语言模型重复采样高效扩展测试时计算量 

**Authors**: Jianhao Chen, Zishuo Xun, Bocheng Zhou, Han Qi, Qiaosheng Zhang, Yang Chen, Wei Hu, Yuzhong Qu, Wanli Ouyang, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00762)  

**Abstract**: This paper presents a simple, effective, and cost-efficient strategy to improve LLM performance by scaling test-time compute. Our strategy builds upon the repeated-sampling-then-voting framework, with a novel twist: incorporating multiple models, even weaker ones, to leverage their complementary strengths that potentially arise from diverse training data and paradigms. By using consistency as a signal, our strategy dynamically switches between models. Theoretical analysis highlights the efficiency and performance advantages of our strategy. Extensive experiments on six datasets demonstrate that our strategy not only outperforms self-consistency and state-of-the-art multi-agent debate approaches, but also significantly reduces inference costs. Additionally, ModelSwitch requires only a few comparable LLMs to achieve optimal performance and can be extended with verification methods, demonstrating the potential of leveraging multiple LLMs in the generation-verification paradigm. 

**Abstract (ZH)**: 本文提出了一种简单、有效且成本效益高的策略，通过扩展测试时计算来提升大语言模型的性能。该策略基于重复采样后再投票的框架，并引入了一个新的元素：结合多个模型，即使是最弱的模型，以利用它们从多样化的训练数据和范式中可能产生的互补优势。通过一致性作为信号，该策略动态切换模型。理论分析突显了该策略的效率和性能优势。在六个数据集上的广泛实验表明，该策略不仅优于自一致性以及最先进的多代理辩论方法，而且显著降低了推理成本。此外，ModelSwitch仅需少量可比的大语言模型即可实现最佳性能，并可通过验证方法扩展，展示了利用多个大语言模型在生成-验证范式中的潜力。 

---
# Personality-Driven Decision-Making in LLM-Based Autonomous Agents 

**Title (ZH)**: 基于LLM的自主代理的人格驱动决策制定 

**Authors**: Lewis Newsham, Daniel Prince  

**Link**: [PDF](https://arxiv.org/pdf/2504.00727)  

**Abstract**: The embedding of Large Language Models (LLMs) into autonomous agents is a rapidly developing field which enables dynamic, configurable behaviours without the need for extensive domain-specific training. In our previous work, we introduced SANDMAN, a Deceptive Agent architecture leveraging the Five-Factor OCEAN personality model, demonstrating that personality induction significantly influences agent task planning. Building on these findings, this study presents a novel method for measuring and evaluating how induced personality traits affect task selection processes - specifically planning, scheduling, and decision-making - in LLM-based agents. Our results reveal distinct task-selection patterns aligned with induced OCEAN attributes, underscoring the feasibility of designing highly plausible Deceptive Agents for proactive cyber defense strategies. 

**Abstract (ZH)**: 大型语言模型嵌入自主代理中的研究：诱导个性特征对基于LLM代理任务选择过程的影响及其在主动网络防御策略中的可行性 

---
# Towards Responsible and Trustworthy Educational Data Mining: Comparing Symbolic, Sub-Symbolic, and Neural-Symbolic AI Methods 

**Title (ZH)**: 负责任且可信赖的教育数据挖掘：符号、亚符号及神经符号AI方法的比较 

**Authors**: Danial Hooshyar, Eve Kikas, Yeongwook Yang, Gustav Šír, Raija Hämäläinen, Tommi Kärkkäinen, Roger Azevedo  

**Link**: [PDF](https://arxiv.org/pdf/2504.00615)  

**Abstract**: Given the demand for responsible and trustworthy AI for education, this study evaluates symbolic, sub-symbolic, and neural-symbolic AI (NSAI) in terms of generalizability and interpretability. Our extensive experiments on balanced and imbalanced self-regulated learning datasets of Estonian primary school students predicting 7th-grade mathematics national test performance showed that symbolic and sub-symbolic methods performed well on balanced data but struggled to identify low performers in imbalanced datasets. Interestingly, symbolic and sub-symbolic methods emphasized different factors in their decision-making: symbolic approaches primarily relied on cognitive and motivational factors, while sub-symbolic methods focused more on cognitive aspects, learned knowledge, and the demographic variable of gender -- yet both largely overlooked metacognitive factors. The NSAI method, on the other hand, showed advantages by: (i) being more generalizable across both classes -- even in imbalanced datasets -- as its symbolic knowledge component compensated for the underrepresented class; and (ii) relying on a more integrated set of factors in its decision-making, including motivation, (meta)cognition, and learned knowledge, thus offering a comprehensive and theoretically grounded interpretability framework. These contrasting findings highlight the need for a holistic comparison of AI methods before drawing conclusions based solely on predictive performance. They also underscore the potential of hybrid, human-centered NSAI methods to address the limitations of other AI families and move us closer to responsible AI for education. Specifically, by enabling stakeholders to contribute to AI design, NSAI aligns learned patterns with theoretical constructs, incorporates factors like motivation and metacognition, and strengthens the trustworthiness and responsibility of educational data mining. 

**Abstract (ZH)**: 负责任和可信赖的AI在教育中的应用：符号、亚符号和神经符号AI的可泛化性和可解释性评价 

---
# LLM-Guided Search for Deletion-Correcting Codes 

**Title (ZH)**: LLM引导的删除纠错码搜索 

**Authors**: Franziska Weindel, Reinhard Heckel  

**Link**: [PDF](https://arxiv.org/pdf/2504.00613)  

**Abstract**: Finding deletion-correcting codes of maximum size has been an open problem for over 70 years, even for a single deletion. In this paper, we propose a novel approach for constructing deletion-correcting codes. A code is a set of sequences satisfying certain constraints, and we construct it by greedily adding the highest-priority sequence according to a priority function. To find good priority functions, we leverage FunSearch, a large language model (LLM)-guided evolutionary search proposed by Romera et al., 2024. FunSearch iteratively generates, evaluates, and refines priority functions to construct large deletion-correcting codes. For a single deletion, our evolutionary search finds functions that construct codes which match known maximum sizes, reach the size of the largest (conjectured optimal) Varshamov-Tenengolts codes where the maximum is unknown, and independently rediscover them in equivalent form. For two deletions, we find functions that construct codes with new best-known sizes for code lengths \( n = 12, 13 \), and \( 16 \), establishing improved lower bounds. These results demonstrate the potential of LLM-guided search for information theory and code design and represent the first application of such methods for constructing error-correcting codes. 

**Abstract (ZH)**: 寻找具有最大规模的删除纠错码的问题已有超过70年未解之谜，即使是单删除的情况亦是如此。在本文中，我们提出了一种新的构建删除纠错码的方法。一种代码是由满足某些约束条件的序列集合组成，我们通过根据优先级函数逐次添加最高优先级的序列来构建它。为了找到良好的优先级函数，我们利用Romera等人于2024年提出的由大语言模型（LLM）引导的进化搜索FunSearch。FunSearch通过迭代生成、评估和改进优先级函数来构建大规模的删除纠错码。对于单删除情况，我们的进化搜索发现的函数构建的码越大，可与已知的最大规模相匹配；对于最大值未知的汉森-特内戈尔茨码，其规模达到已知最大（猜想最优）的规模；并且能够独立地以等效形式重新发现它们。对于双删除情况，我们发现的函数构建的码长度分别为\( n = 12, 13 \)和\( 16 \)时，具有新的最佳已知规模，从而建立了改进的下界。这些结果展示了LLM引导的搜索在信息理论和码设计中的潜力，并代表了此类方法首次用于构建纠错码的应用。 

---
# Recitation over Reasoning: How Cutting-Edge Language Models Can Fail on Elementary School-Level Reasoning Problems? 

**Title (ZH)**: Recitation over Reasoning: How Cutting-Edge Language Models Can Fail on Elementary School-Level Reasoning Problems? 

**Authors**: Kai Yan, Yufei Xu, Zhengyin Du, Xuesong Yao, Zheyu Wang, Xiaowen Guo, Jiecao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.00509)  

**Abstract**: The rapid escalation from elementary school-level to frontier problems of the difficulty for LLM benchmarks in recent years have weaved a miracle for researchers that we are only inches away from surpassing human intelligence. However, is the LLMs' remarkable reasoning ability indeed comes from true intelligence by human standards, or are they simply reciting solutions witnessed during training at an Internet level? To study this problem, we propose RoR-Bench, a novel, multi-modal benchmark for detecting LLM's recitation behavior when asked simple reasoning problems but with conditions subtly shifted, and conduct empirical analysis on our benchmark. Surprisingly, we found existing cutting-edge LLMs unanimously exhibits extremely severe recitation behavior; by changing one phrase in the condition, top models such as OpenAI-o1 and DeepSeek-R1 can suffer $60\%$ performance loss on elementary school-level arithmetic and reasoning problems. Such findings are a wake-up call to the LLM community that compels us to re-evaluate the true intelligence level of cutting-edge LLMs. 

**Abstract (ZH)**: 近年来，大语言模型基准从小学级问题到前沿问题难度的迅速升级，为研究人员编织了一个奇迹般的前景，即我们几乎达到了超越人类智能的水平。然而，大语言模型卓越的推理能力究竟是基于人类标准的真正智能，还是仅仅在网上训练时回忆并重现的解决方案？为研究这一问题，我们提出了RoR-Bench这一新颖的多模态基准，用于检测大语言模型在回答简单推理问题但在条件上细微变化时的回忆行为，并对基准进行了实证分析。令人惊讶的是，我们发现现有最先进的大语言模型普遍表现出极其严重的回忆行为；仅通过改变一个条件短语，如OpenAI-o1和DeepSeek-R1等顶级模型在小学级算术和推理问题上的性能损失高达60%。这一发现是对大语言模型社区的警醒，促使我们重新评估当前最先进大语言模型的真实智能水平。 

---
# Hawkeye:Efficient Reasoning with Model Collaboration 

**Title (ZH)**: 慧眼：模型协作的高效推理 

**Authors**: Jianshu She, Zhuohao Li, Zhemin Huang, Qi Li, Peiran Xu, Haonan Li, Qirong Ho  

**Link**: [PDF](https://arxiv.org/pdf/2504.00424)  

**Abstract**: Chain-of-Thought (CoT) reasoning has demonstrated remarkable effectiveness in enhancing the reasoning abilities of large language models (LLMs). However, its efficiency remains a challenge due to the generation of excessive intermediate reasoning tokens, which introduce semantic redundancy and overly detailed reasoning steps. Moreover, computational expense and latency are significant concerns, as the cost scales with the number of output tokens, including those intermediate steps. In this work, we observe that most CoT tokens are unnecessary, and retaining only a small portion of them is sufficient for producing high-quality responses. Inspired by this, we propose HAWKEYE, a novel post-training and inference framework where a large model produces concise CoT instructions to guide a smaller model in response generation. HAWKEYE quantifies redundancy in CoT reasoning and distills high-density information via reinforcement learning. By leveraging these concise CoTs, HAWKEYE is able to expand responses while reducing token usage and computational cost significantly. Our evaluation shows that HAWKEYE can achieve comparable response quality using only 35% of the full CoTs, while improving clarity, coherence, and conciseness by approximately 10%. Furthermore, HAWKEYE can accelerate end-to-end reasoning by up to 3.4x on complex math tasks while reducing inference cost by up to 60%. HAWKEYE will be open-sourced and the models will be available soon. 

**Abstract (ZH)**: HAWKEYE：一种新型的后训练和推理框架，用于精简链式推理指令以提高生成质量并加速推理 

---
# CyberBOT: Towards Reliable Cybersecurity Education via Ontology-Grounded Retrieval Augmented Generation 

**Title (ZH)**: CyberBOT: 基于本体导向检索增强生成的可靠网络安全教育探索 

**Authors**: Chengshuai Zhao, Riccardo De Maria, Tharindu Kumarage, Kumar Satvik Chaudhary, Garima Agrawal, Yiwen Li, Jongchan Park, Yuli Deng, Ying-Chih Chen, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00389)  

**Abstract**: Advancements in large language models (LLMs) have enabled the development of intelligent educational tools that support inquiry-based learning across technical domains. In cybersecurity education, where accuracy and safety are paramount, systems must go beyond surface-level relevance to provide information that is both trustworthy and domain-appropriate. To address this challenge, we introduce CyberBOT, a question-answering chatbot that leverages a retrieval-augmented generation (RAG) pipeline to incorporate contextual information from course-specific materials and validate responses using a domain-specific cybersecurity ontology. The ontology serves as a structured reasoning layer that constrains and verifies LLM-generated answers, reducing the risk of misleading or unsafe guidance. CyberBOT has been deployed in a large graduate-level course at Arizona State University (ASU), where more than one hundred students actively engage with the system through a dedicated web-based platform. Computational evaluations in lab environments highlight the potential capacity of CyberBOT, and a forthcoming field study will evaluate its pedagogical impact. By integrating structured domain reasoning with modern generative capabilities, CyberBOT illustrates a promising direction for developing reliable and curriculum-aligned AI applications in specialized educational contexts. 

**Abstract (ZH)**: 大型语言模型的进步使开发支持基于探究的学习的智能教育工具成为可能，这些工具适用于各类技术领域。在信息安全教育中，由于准确性与安全性至关重要，系统必须超越表面相关性，提供既可信又符合特定领域的信息。为应对这一挑战，我们引入了CyberBOT，这是一种利用检索增强生成（RAG）pipeline的问答聊天机器人，它可以整合课程特定材料的上下文信息，并使用特定领域的安全Ontology验证响应。此Ontology作为结构化推理层，限制并验证由大型语言模型生成的答案，从而降低误导性或不安全指导的风险。CyberBOT已在亚利桑那州立大学（ASU）的一门大型研究生课程中部署，超过百名学生通过专用的WEB平台积极与该系统互动。实验室环境下的计算评估突显了CyberBOT的潜在容量，而即将进行的实地研究将评估其教学影响。通过将结构化领域推理与现代生成能力相结合，CyberBOT展示了在特定教育环境中开发可靠且符合课程要求的AI应用的一个有前景的方向。 

---
# Collaborative LLM Numerical Reasoning with Local Data Protection 

**Title (ZH)**: 本地数据保护下的协作大语言模型数值推理 

**Authors**: Min Zhang, Yuzhe Lu, Yun Zhou, Panpan Xu, Lin Lee Cheong, Chang-Tien Lu, Haozhu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00299)  

**Abstract**: Numerical reasoning over documents, which demands both contextual understanding and logical inference, is challenging for low-capacity local models deployed on computation-constrained devices. Although such complex reasoning queries could be routed to powerful remote models like GPT-4, exposing local data raises significant data leakage concerns. Existing mitigation methods generate problem descriptions or examples for remote assistance. However, the inherent complexity of numerical reasoning hinders the local model from generating logically equivalent queries and accurately inferring answers with remote guidance. In this paper, we present a model collaboration framework with two key innovations: (1) a context-aware synthesis strategy that shifts the query domains while preserving logical consistency; and (2) a tool-based answer reconstruction approach that reuses the remote-generated problem-solving pattern with code snippets. Experimental results demonstrate that our method achieves better reasoning accuracy than solely using local models while providing stronger data protection than fully relying on remote models. Furthermore, our method improves accuracy by 16.2% - 43.6% while reducing data leakage by 2.3% - 44.6% compared to existing data protection approaches. 

**Abstract (ZH)**: 基于文档的数值推理需要同时具备上下文理解与逻辑推理能力，对于部署在计算受限设备上的低容量本地模型来说是一个挑战。虽然这类复杂的推理查询可以路由到强大的远程模型（如GPT-4）进行处理，但将本地数据暴露出去引发了显著的数据泄露问题。现有缓解方法生成问题描述或示例以寻求远程协助。然而，数值推理的固有复杂性阻碍了本地模型生成逻辑等价的查询以及在远程指导下的准确推理。在本文中，我们提出了一种模型协作框架，包含两个关键创新：（1）一种上下文感知的合成策略，能够在保持逻辑一致性的同时转换查询域；（2）一种基于工具的答案重构方法，该方法利用代码片段重用远程生成的解决问题模式。实验结果表明，与仅使用本地模型相比，我们的方法在推理准确性上表现更好，同时在数据保护强度上优于完全依赖远程模型的方法。此外，与现有数据保护方法相比，我们的方法将准确性提高了16.2% - 43.6%，同时将数据泄露减少了2.3% - 44.6%。 

---
# Exploration and Adaptation in Non-Stationary Tasks with Diffusion Policies 

**Title (ZH)**: 非稳态任务中扩散策略的探索与适应 

**Authors**: Gunbir Singh Baveja  

**Link**: [PDF](https://arxiv.org/pdf/2504.00280)  

**Abstract**: This paper investigates the application of Diffusion Policy in non-stationary, vision-based RL settings, specifically targeting environments where task dynamics and objectives evolve over time. Our work is grounded in practical challenges encountered in dynamic real-world scenarios such as robotics assembly lines and autonomous navigation, where agents must adapt control strategies from high-dimensional visual inputs. We apply Diffusion Policy -- which leverages iterative stochastic denoising to refine latent action representations-to benchmark environments including Procgen and PointMaze. Our experiments demonstrate that, despite increased computational demands, Diffusion Policy consistently outperforms standard RL methods such as PPO and DQN, achieving higher mean and maximum rewards with reduced variability. These findings underscore the approach's capability to generate coherent, contextually relevant action sequences in continuously shifting conditions, while also highlighting areas for further improvement in handling extreme non-stationarity. 

**Abstract (ZH)**: 本文研究了扩散策略在非站稳态、基于视觉的强化学习环境中的应用，特别是针对任务动力学和目标随时间演变的环境。我们的工作基于在动态现实场景中如机器人装配线和自主导航中遇到的实用挑战，其中代理必须从高维视觉输入中适应控制策略。我们应用扩散策略——利用迭代随机降噪来细化潜在动作表示——在Procgen和PointMaze等基准环境中进行了测试。实验结果表明，尽管增加了计算需求，扩散策略在平均奖励和最大奖励方面始终优于标准的RL方法如PPO和DQN，并且具有更低的奖励变异性。这些发现突显了该方法在连续变化条件下生成连贯且上下文相关动作序列的能力，同时也指出了在处理极端非站稳态方面需要改进的领域。 

---
# Rack Position Optimization in Large-Scale Heterogeneous Data Centers 

**Title (ZH)**: 大规模异构数据中心机架位置优化 

**Authors**: Chang-Lin Chen, Jiayu Chen, Tian Lan, Zhaoxia Zhao, Hongbo Dong, Vaneet Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2504.00277)  

**Abstract**: As rapidly growing AI computational demands accelerate the need for new hardware installation and maintenance, this work explores optimal data center resource management by balancing operational efficiency with fault tolerance through strategic rack positioning considering diverse resources and locations. Traditional mixed-integer programming (MIP) approaches often struggle with scalability, while heuristic methods may result in significant sub-optimality. To address these issues, this paper presents a novel two-tier optimization framework using a high-level deep reinforcement learning (DRL) model to guide a low-level gradient-based heuristic for local search. The high-level DRL agent employs Leader Reward for optimal rack type ordering, and the low-level heuristic efficiently maps racks to positions, minimizing movement counts and ensuring fault-tolerant resource distribution. This approach allows scalability to over 100,000 positions and 100 rack types. Our method outperformed the gradient-based heuristic by 7\% on average and the MIP solver by over 30\% in objective value. It achieved a 100\% success rate versus MIP's 97.5\% (within a 20-minute limit), completing in just 2 minutes compared to MIP's 1630 minutes (i.e., almost 4 orders of magnitude improvement). Unlike the MIP solver, which showed performance variability under time constraints and high penalties, our algorithm consistently delivered stable, efficient results - an essential feature for large-scale data center management. 

**Abstract (ZH)**: 面向新型计算需求的数据中心资源管理优化：基于高阶深度强化学习的两层优化框架 

---
# Large Language Models in Numberland: A Quick Test of Their Numerical Reasoning Abilities 

**Title (ZH)**: 大语言模型在数 land 的快速测试：它们的数值推理能力 

**Authors**: Roussel Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2504.00226)  

**Abstract**: An essential element of human mathematical reasoning is our number sense -- an abstract understanding of numbers and their relationships -- which allows us to solve problems involving vast number spaces using limited computational resources. Mathematical reasoning of Large Language Models (LLMs) is often tested on high-level problems (such as Olympiad challenges, geometry, word problems, and puzzles), but their low-level number sense remains less explored. We introduce "Numberland," a 100-problem test to evaluate the numerical reasoning abilities of LLM-based agents. The tasks -- basic operations, advanced calculations (e.g., exponentiation, complex numbers), prime number checks, and the 24 game -- aim to test elementary skills and their integration in solving complex and uncertain problems. We evaluated five LLM-based agents: OpenAI's o1 and o1-mini, Google Gemini, Microsoft Copilot, and Anthropic Claude. They scored 74-95% on the first three tasks that allow deterministic steps to solutions. In the 24 game, which needs trial-and-error search, performance dropped to 10-73%. We tested the top 24 solver (o1 with 73% accuracy) on 25 harder problems, and its score fell to 27%, confirming search as a bottleneck. These results, along with the types of mistakes, suggest a fragile number of LLMs, which is a bit surprising given their prowess in challenging benchmarks. The limits of LLM numerical reasoning highlight the scope of simple, targeted tests to evaluate and explain LLM math skills to ensure safe use. 

**Abstract (ZH)**: 一种人类数学推理的关键元素是我们的数感——一种对数字及其关系的抽象理解——它允许我们在有限的计算资源下解决涉及巨大数字空间的问题。大规模语言模型（LLMs）的数学推理经常通过奥林匹克挑战、几何学、文字问题和谜题等高级问题进行测试，但其低级数感的研究相对较少。我们引入了“Numberland”，一个包含100个问题的测试，以评估基于LLM的代理的数值推理能力。任务包括基本操作、高级计算（如幂运算、复数）、素数检验和24点游戏，旨在测试基础技能及其在解决复杂和不确定问题中的整合。我们评估了五种基于LLM的代理：OpenAI的o1和o1-mini、Google Gemini、Microsoft Copilot和Anthropic Claude。它们在前三项允许确定步骤的问题上获得了74-95%的分数。在需要试错搜索的24点游戏中，性能下降到10-73%。我们测试了最准确的24点游戏的求解器（o1，准确率为73%）在25个更难的问题上，其得分降至27%，确认了搜索是一个瓶颈。这些结果以及错误类型表明，LLMs在数值方面的脆弱性，这在考虑到它们在具有挑战性的基准测试中的表现时有些出人意料。LLMs数值推理的局限性强调了对简单和针对性测试的需求，以评估和解释LLMs的数学技能，确保其安全使用。 

---
# LLMs for Explainable AI: A Comprehensive Survey 

**Title (ZH)**: 可解释AI中的LLMs综述：一项全面调研 

**Authors**: Ahsan Bilal, David Ebert, Beiyu Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.00125)  

**Abstract**: Large Language Models (LLMs) offer a promising approach to enhancing Explainable AI (XAI) by transforming complex machine learning outputs into easy-to-understand narratives, making model predictions more accessible to users, and helping bridge the gap between sophisticated model behavior and human interpretability. AI models, such as state-of-the-art neural networks and deep learning models, are often seen as "black boxes" due to a lack of transparency. As users cannot fully understand how the models reach conclusions, users have difficulty trusting decisions from AI models, which leads to less effective decision-making processes, reduced accountabilities, and unclear potential biases. A challenge arises in developing explainable AI (XAI) models to gain users' trust and provide insights into how models generate their outputs. With the development of Large Language Models, we want to explore the possibilities of using human language-based models, LLMs, for model explainabilities. This survey provides a comprehensive overview of existing approaches regarding LLMs for XAI, and evaluation techniques for LLM-generated explanation, discusses the corresponding challenges and limitations, and examines real-world applications. Finally, we discuss future directions by emphasizing the need for more interpretable, automated, user-centric, and multidisciplinary approaches for XAI via LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）为增强可解释人工智能（XAI）提供了有前途的方法，通过将复杂的机器学习输出转换为易于理解的故事，使模型预测更加易于用户的理解，并有助于弥合高级模型行为与人类可解释性之间的差距。由于透明度不足，AI模型，如最先进的神经网络和深度学习模型，常被视为“黑箱”。由于用户无法完全理解模型如何得出结论，用户难以信任AI模型的决策，从而导致决策过程效率降低、责任减少以及潜在偏见不清晰。在开发可解释AI（XAI）模型以赢得用户信任并提供有关模型生成输出的见解方面存在挑战。随着大型语言模型的发展，我们旨在探索使用基于人类语言的模型—LLMs—提高模型可解释性的可能性。本文综述了现有用于XAI的LLM方法及其评估技术，讨论了相应挑战和限制，并分析了实际应用。最后，我们强调通过LLMs实现XAI的更可解释性、自动化、用户中心和跨学科方法的重要性，指出未来的研究方向。 

---
# The Axiom-Based Atlas: A Structural Mapping of Theorems via Foundational Proof Vectors 

**Title (ZH)**: 基于公理的图谱：通过基础证明向量的定理结构映射 

**Authors**: Harim Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2504.00063)  

**Abstract**: The Axiom-Based Atlas is a novel framework that structurally represents mathematical theorems as proof vectors over foundational axiom systems. By mapping the logical dependencies of theorems onto vectors indexed by axioms - such as those from Hilbert geometry, Peano arithmetic, or ZFC - we offer a new way to visualize, compare, and analyze mathematical knowledge. This vector-based formalism not only captures the logical foundation of theorems but also enables quantitative similarity metrics - such as cosine distance - between mathematical results, offering a new analytic layer for structural comparison. Using heatmaps, vector clustering, and AI-assisted modeling, this atlas enables the grouping of theorems by logical structure, not just by mathematical domain. We also introduce a prototype assistant (Atlas-GPT) that interprets natural language theorems and suggests likely proof vectors, supporting future applications in automated reasoning, mathematical education, and formal verification.
This direction is partially inspired by Terence Tao's recent reflections on the convergence of symbolic and structural mathematics. The Axiom-Based Atlas aims to provide a scalable, interpretable model of mathematical reasoning that is both human-readable and AI-compatible, contributing to the future landscape of formal mathematical systems. 

**Abstract (ZH)**: 基于公理的图谱：一种将数学定理结构化表示为公理基础公理系统上的证明向量的新框架。 

---
# GeometryCrafter: Consistent Geometry Estimation for Open-world Videos with Diffusion Priors 

**Title (ZH)**: GeometryCrafter: 开放世界视频中的扩散先验一致几何估计 

**Authors**: Tian-Xing Xu, Xiangjun Gao, Wenbo Hu, Xiaoyu Li, Song-Hai Zhang, Ying Shan  

**Link**: [PDF](https://arxiv.org/pdf/2504.01016)  

**Abstract**: Despite remarkable advancements in video depth estimation, existing methods exhibit inherent limitations in achieving geometric fidelity through the affine-invariant predictions, limiting their applicability in reconstruction and other metrically grounded downstream tasks. We propose GeometryCrafter, a novel framework that recovers high-fidelity point map sequences with temporal coherence from open-world videos, enabling accurate 3D/4D reconstruction, camera parameter estimation, and other depth-based applications. At the core of our approach lies a point map Variational Autoencoder (VAE) that learns a latent space agnostic to video latent distributions for effective point map encoding and decoding. Leveraging the VAE, we train a video diffusion model to model the distribution of point map sequences conditioned on the input videos. Extensive evaluations on diverse datasets demonstrate that GeometryCrafter achieves state-of-the-art 3D accuracy, temporal consistency, and generalization capability. 

**Abstract (ZH)**: 尽管在视频深度估计方面取得了显著进展，现有方法通过仿射不变预测在实现几何保真方面仍存在固有限制，限制了其在重建及其他基于度量的下游任务中的应用。我们提出了一种名为GeometryCrafter的新框架，该框架能够从开放世界视频中恢复具有时间连续性的高保真点图序列，从而实现精确的3D/4D重建、相机参数估计及其他基于深度的应用。该方法的核心是一种点图变分自编码器（VAE），能够学习与视频潜在分布无关的潜在空间，从而有效进行点图编码和解码。利用VAE，我们训练了一个视频扩散模型，该模型能够基于输入视频对点图序列的概率分布进行建模。在多种数据集上的 extensive 评估表明，GeometryCrafter 实现了最先进的3D准确性、时间一致性及泛化能力。 

---
# IntrinsiX: High-Quality PBR Generation using Image Priors 

**Title (ZH)**: IntrinsiX: 使用图像先验的高质量物理基于渲染生成 

**Authors**: Peter Kocsis, Lukas Höllein, Matthias Nießner  

**Link**: [PDF](https://arxiv.org/pdf/2504.01008)  

**Abstract**: We introduce IntrinsiX, a novel method that generates high-quality intrinsic images from text description. In contrast to existing text-to-image models whose outputs contain baked-in scene lighting, our approach predicts physically-based rendering (PBR) maps. This enables the generated outputs to be used for content creation scenarios in core graphics applications that facilitate re-lighting, editing, and texture generation tasks. In order to train our generator, we exploit strong image priors, and pre-train separate models for each PBR material component (albedo, roughness, metallic, normals). We then align these models with a new cross-intrinsic attention formulation that concatenates key and value features in a consistent fashion. This allows us to exchange information between each output modality and to obtain semantically coherent PBR predictions. To ground each intrinsic component, we propose a rendering loss which provides image-space signals to constrain the model, thus facilitating sharp details also in the output BRDF properties. Our results demonstrate detailed intrinsic generation with strong generalization capabilities that outperforms existing intrinsic image decomposition methods used with generated images by a significant margin. Finally, we show a series of applications, including re-lighting, editing, and text-conditioned room-scale PBR texture generation. 

**Abstract (ZH)**: 我们引入了IntrinsiX，一种新颖的方法，能够从文本描述生成高质量的内在图像。与现有包含内置场景照明的文本到图像模型不同，我们的方法预测基于物理的渲染（PBR）图。这使得生成的输出能够用于核心图形应用程序中的内容创作场景，方便重新照明、编辑和纹理生成任务。为了训练我们的生成器，我们利用强大的图像先验知识，并为每种PBR材质成分（反射率、粗糙度、金属度、法线）分别预训练模型。然后，我们通过一种新的跨内在注意力形式化将这些模型与一致的关键特征和价值特征连接起来。这允许我们在每个输出模态之间交换信息，并获得语义上一致的PBR预测。为了使每个内在成分落地，我们提出了一个渲染损失，提供了图像空间信号来约束模型，从而在输出BRDF属性中实现清晰的细节。我们的结果展示了详细而具有强泛化能力的内在生成，明显优于现有用于生成图像的内在图像分解方法。最后，我们展示了包括重新照明、编辑以及条件文本的房间尺度PBR纹理生成等一系列应用。 

---
# When To Solve, When To Verify: Compute-Optimal Problem Solving and Generative Verification for LLM Reasoning 

**Title (ZH)**: 何时求解，何时验证：面向LLM推理的计算最优问题求解与生成验证 

**Authors**: Nishad Singhi, Hritik Bansal, Arian Hosseini, Aditya Grover, Kai-Wei Chang, Marcus Rohrbach, Anna Rohrbach  

**Link**: [PDF](https://arxiv.org/pdf/2504.01005)  

**Abstract**: Scaling test-time compute has emerged as a key strategy for enhancing the reasoning capabilities of large language models (LLMs), particularly in tasks like mathematical problem-solving. A traditional approach, Self-Consistency (SC), generates multiple solutions to a problem and selects the most common answer via majority voting. Another common method involves scoring each solution with a reward model (verifier) and choosing the best one. Recent advancements in Generative Reward Models (GenRM) reframe verification as a next-token prediction task, enabling inference-time scaling along a new axis. Specifically, GenRM generates multiple verification chains-of-thought to score each solution. Under a limited inference budget, this introduces a fundamental trade-off: should you spend the budget on scaling solutions via SC or generate fewer solutions and allocate compute to verification via GenRM? To address this, we evaluate GenRM against SC under a fixed inference budget. Interestingly, we find that SC is more compute-efficient than GenRM for most practical inference budgets across diverse models and datasets. For instance, GenRM first matches SC after consuming up to 8x the inference compute and requires significantly more compute to outperform it. Furthermore, we derive inference scaling laws for the GenRM paradigm, revealing that compute-optimal inference favors scaling solution generation more aggressively than scaling the number of verifications. Our work provides practical guidance on optimizing test-time scaling by balancing solution generation and verification. The code is available at this https URL. 

**Abstract (ZH)**: 扩展测试时计算量已成为增强大型语言模型（LLMs）推理能力的关键策略，特别是在数学问题求解等任务中。传统的自一致性（SC）方法生成多个问题解决方案，并通过 majority voting 选择最常见的答案。另一种常见方法是对每个解决方案打分（使用验证器），并选择得分最高的。最新的生成奖励模型（GenRM）将验证重新构想为下一个词预测任务，从而可以在新的轴上扩展推理。具体而言，GenRM 为每个解决方案生成多个验证链式思考来评分。在有限的推理预算下，这引入了一个基本的权衡：你是否应该将预算用于通过 SC 扩展解决方案的数量，还是生成更少的解决方案并为验证分配计算资源通过 GenRM？为了解决这个问题，我们在固定推理预算下评估了 GenRM 对 SC 的表现。有趣的是，我们发现，在大多数实际推理预算下，SC 在各种模型和数据集上的计算效率高于 GenRM。例如，GenRM 需要消耗高达 8 倍的推理计算量才能与 SC 相当，并且需要显著更多的计算量才能优于 SC。此外，我们推导了 GenRM 模式下的推理扩展法则，发现计算最优的推理倾向于更激进地扩展解决方案生成而不是扩展验证的数量。我们的工作提供了优化测试时扩展的实际指导，通过平衡解决方案生成和验证。代码可在以下链接获取。 

---
# Token embeddings violate the manifold hypothesis 

**Title (ZH)**: Token嵌入违反流形假说 

**Authors**: Michael Robinson, Sourya Dey, Tony Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.01002)  

**Abstract**: To fully understand the behavior of a large language model (LLM) requires our understanding of its input space. If this input space differs from our assumption, our understanding of and conclusions about the LLM is likely flawed, regardless of its architecture. Here, we elucidate the structure of the token embeddings, the input domain for LLMs, both empirically and theoretically. We present a generalized and statistically testable model where the neighborhood of each token splits into well-defined signal and noise dimensions.
This model is based on a generalization of a manifold called a fiber bundle, so we denote our hypothesis test as the ``fiber bundle null.'' Failing to reject the null is uninformative, but rejecting it at a specific token indicates that token has a statistically significant local structure, and so is of interest to us. By running our test over several open-source LLMs, each with unique token embeddings, we find that the null is frequently rejected, and so the token subspace is provably not a fiber bundle and hence also not a manifold. As a consequence of our findings, when an LLM is presented with two semantically equivalent prompts, and if one prompt contains a token implicated by our test, that prompt will likely exhibit more output variability proportional to the local signal dimension of the token. 

**Abstract (ZH)**: 要全面理解大型语言模型（LLM）的行为，需要我们理解其输入空间。如果这个输入空间与我们的假设不同，那么我们对LLM的理解和结论很可能是有缺陷的，无论其架构如何。在这里，我们详述了token嵌入的结构，即LLM的输入域，通过实证和理论的方法。我们提出了一个一般化且可统计检验的模型，其中每个token的邻域被分为明确的信号维度和噪声维度。

该模型基于纤维丛这一流形的一般化概念，因此我们将我们的假设检验称为“纤维丛零假设”。未能拒绝零假设是无信息性的，但特定token处拒绝零假设表明该token具有统计显著的局部结构，从而引起我们的兴趣。通过在多个开源LLM上运行我们的测试，每种LLM具有独特的token嵌入，我们发现零假设经常被拒绝，因此token子空间不是纤维丛，也不一定是流形。据此，当LLM接收到两个语义等价的提示时，如果一个提示包含我们的测试所指的token，则该提示的输出变异性很可能会与该token的局部信号维度成比例。 

---
# Zero-shot Benchmarking: A Framework for Flexible and Scalable Automatic Evaluation of Language Models 

**Title (ZH)**: 零样本基准测试：一种灵活可扩展的语言模型自动评估框架 

**Authors**: José Pombal, Nuno M. Guerreiro, Ricardo Rei, André F. T. Martins  

**Link**: [PDF](https://arxiv.org/pdf/2504.01001)  

**Abstract**: As language models improve and become capable of performing more complex tasks across modalities, evaluating them automatically becomes increasingly challenging. Developing strong and robust task-specific automatic metrics gets harder, and human-annotated test sets -- which are expensive to create -- saturate more quickly. A compelling alternative is to design reliable strategies to automate the creation of test data and evaluation, but previous attempts either rely on pre-existing data, or focus solely on individual tasks. We present Zero-shot Benchmarking (ZSB), a framework for creating high-quality benchmarks for any task by leveraging language models for both synthetic test data creation and evaluation. ZSB is simple and flexible: it requires only the creation of a prompt for data generation and one for evaluation; it is scalable to tasks and languages where collecting real-world data is costly or impractical; it is model-agnostic, allowing the creation of increasingly challenging benchmarks as models improve. To assess the effectiveness of our framework, we create benchmarks for five text-only tasks and a multi-modal one: general capabilities in four languages (English, Chinese, French, and Korean), translation, and general vision-language capabilities in English. We then rank a broad range of open and closed systems on our benchmarks. ZSB rankings consistently correlate strongly with human rankings, outperforming widely-adopted standard benchmarks. Through ablations, we find that strong benchmarks can be created with open models, and that judge model size and dataset variety are crucial drivers of performance. We release all our benchmarks, and code to reproduce our experiments and to produce new benchmarks. 

**Abstract (ZH)**: 基于语言模型的零样本基准测试框架：创建多模态任务的高质量基准 

---
# MergeVQ: A Unified Framework for Visual Generation and Representation with Disentangled Token Merging and Quantization 

**Title (ZH)**: MergeVQ：一种基于解耦令牌合并与量化的一体化视觉生成与表示框架 

**Authors**: Siyuan Li, Luyuan Zhang, Zedong Wang, Juanxi Tian, Cheng Tan, Zicheng Liu, Chang Yu, Qingsong Xie, Haonan Lu, Haoqian Wang, Zhen Lei  

**Link**: [PDF](https://arxiv.org/pdf/2504.00999)  

**Abstract**: Masked Image Modeling (MIM) with Vector Quantization (VQ) has achieved great success in both self-supervised pre-training and image generation. However, most existing methods struggle to address the trade-off in shared latent space for generation quality vs. representation learning and efficiency. To push the limits of this paradigm, we propose MergeVQ, which incorporates token merging techniques into VQ-based generative models to bridge the gap between image generation and visual representation learning in a unified architecture. During pre-training, MergeVQ decouples top-k semantics from latent space with the token merge module after self-attention blocks in the encoder for subsequent Look-up Free Quantization (LFQ) and global alignment and recovers their fine-grained details through cross-attention in the decoder for reconstruction. As for the second-stage generation, we introduce MergeAR, which performs KV Cache compression for efficient raster-order prediction. Extensive experiments on ImageNet verify that MergeVQ as an AR generative model achieves competitive performance in both visual representation learning and image generation tasks while maintaining favorable token efficiency and inference speed. The code and model will be available at this https URL. 

**Abstract (ZH)**: 基于向量量化（VQ）的掩码图像建模（MIM）在自我监督预训练和图像生成方面取得了巨大成功。然而，大多数现有方法在生成质量与表示学习及效率之间的权衡中表现不佳。为推动这一范式的极限，我们提出 MergeVQ，将token压缩技术集成到基于向量量化的生成模型中，以统一架构在图像生成和视觉表示学习之间搭建桥梁。在预训练阶段，MergeVQ 在编码器自注意力模块之后通过token合并模块解耦 top-k 语义，并在后续的无查找量化（LFQ）和全局对齐中保持这些语义，然后通过解码器中的交叉注意力恢复其细颗粒度细节进行重构。对于生成的第二阶段，我们引入 MergeAR，通过KV缓存压缩提高按扫描线顺序预测的效率。在ImageNet上的广泛实验验证了MergeVQ作为AR生成模型在视觉表示学习和图像生成任务中的竞争力，同时保持了有利的token效率和推理速度。相关代码和模型将在此处提供。 

---
# MedReason: Eliciting Factual Medical Reasoning Steps in LLMs via Knowledge Graphs 

**Title (ZH)**: MedReason: 通过知识图谱提取LLM中的事实医学推理步骤 

**Authors**: Juncheng Wu, Wenlong Deng, Xingxuan Li, Sheng Liu, Taomian Mi, Yifan Peng, Ziyang Xu, Yi Liu, Hyunjin Cho, Chang-In Choi, Yihan Cao, Hui Ren, Xiang Li, Xiaoxiao Li, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.00993)  

**Abstract**: Medical tasks such as diagnosis and treatment planning require precise and complex reasoning, particularly in life-critical domains. Unlike mathematical reasoning, medical reasoning demands meticulous, verifiable thought processes to ensure reliability and accuracy. However, there is a notable lack of datasets that provide transparent, step-by-step reasoning to validate and enhance the medical reasoning ability of AI models. To bridge this gap, we introduce MedReason, a large-scale high-quality medical reasoning dataset designed to enable faithful and explainable medical problem-solving in large language models (LLMs). We utilize a structured medical knowledge graph (KG) to convert clinical QA pairs into logical chains of reasoning, or ``thinking paths'', which trace connections from question elements to answers via relevant KG entities. Each path is validated for consistency with clinical logic and evidence-based medicine. Our pipeline generates detailed reasoning for various medical questions from 7 medical datasets, resulting in a dataset of 32,682 question-answer pairs, each with detailed, step-by-step explanations. Experiments demonstrate that fine-tuning with our dataset consistently boosts medical problem-solving capabilities, achieving significant gains of up to 7.7% for DeepSeek-Ditill-8B. Our top-performing model, MedReason-8B, outperforms the Huatuo-o1-8B, a state-of-the-art medical reasoning model, by up to 4.2% on the clinical benchmark MedBullets. We also engage medical professionals from diverse specialties to assess our dataset's quality, ensuring MedReason offers accurate and coherent medical reasoning. Our data, models, and code will be publicly available. 

**Abstract (ZH)**: 大规模高质量医学推理数据集MedReason：提升大型语言模型的医学问题解决能力 

---
# Accelerating drug discovery with Artificial: a whole-lab orchestration and scheduling system for self-driving labs 

**Title (ZH)**: 使用人工智能加速药物发现：一个自动化实验室全流程调度系统 

**Authors**: Yao Fehlis, Paul Mandel, Charles Crain, Betty Liu, David Fuller  

**Link**: [PDF](https://arxiv.org/pdf/2504.00986)  

**Abstract**: Self-driving labs are transforming drug discovery by enabling automated, AI-guided experimentation, but they face challenges in orchestrating complex workflows, integrating diverse instruments and AI models, and managing data efficiently. Artificial addresses these issues with a comprehensive orchestration and scheduling system that unifies lab operations, automates workflows, and integrates AI-driven decision-making. By incorporating AI/ML models like NVIDIA BioNeMo - which facilitates molecular interaction prediction and biomolecular analysis - Artificial enhances drug discovery and accelerates data-driven research. Through real-time coordination of instruments, robots, and personnel, the platform streamlines experiments, enhances reproducibility, and advances drug discovery. 

**Abstract (ZH)**: 自驱动实验室正在通过实现自动化的、AI引导的实验来转变药物发现，但面临复杂工作流调度、多元仪器和AI模型集成以及数据管理的挑战。Artificial通过一个全面的调度和编排系统解决了这些问题，统一了实验室运营、自动化工作流和AI驱动的决策集成，提升了药物发现能力并加速了数据驱动的研究。通过实时协调仪器、机器人和人员，该平台简化了实验流程，提高了实验的可重复性，并推动了药物发现的进步。 

---
# WorldScore: A Unified Evaluation Benchmark for World Generation 

**Title (ZH)**: 世界生成统一评估基准：WorldScore 

**Authors**: Haoyi Duan, Hong-Xing Yu, Sirui Chen, Li Fei-Fei, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00983)  

**Abstract**: We introduce the WorldScore benchmark, the first unified benchmark for world generation. We decompose world generation into a sequence of next-scene generation tasks with explicit camera trajectory-based layout specifications, enabling unified evaluation of diverse approaches from 3D and 4D scene generation to video generation models. The WorldScore benchmark encompasses a curated dataset of 3,000 test examples that span diverse worlds: static and dynamic, indoor and outdoor, photorealistic and stylized. The WorldScore metrics evaluate generated worlds through three key aspects: controllability, quality, and dynamics. Through extensive evaluation of 19 representative models, including both open-source and closed-source ones, we reveal key insights and challenges for each category of models. Our dataset, evaluation code, and leaderboard can be found at this https URL 

**Abstract (ZH)**: 世界评分基准：首个统一的世界生成基准 

---
# Resource Allocation for RIS-Assisted CoMP-NOMA Networks using Reinforcement Learning 

**Title (ZH)**: RIS辅助协作多点传输和非正交多址网络的资源分配方法研究（使用强化学习） 

**Authors**: Muhammad Umer, Muhammad Ahmed Mohsin, Huma Ghafoor, Syed Ali Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00975)  

**Abstract**: This thesis delves into the forefront of wireless communication by exploring the synergistic integration of three transformative technologies: STAR-RIS, CoMP, and NOMA. Driven by the ever-increasing demand for higher data rates, improved spectral efficiency, and expanded coverage in the evolving landscape of 6G development, this research investigates the potential of these technologies to revolutionize future wireless networks.
The thesis analyzes the performance gains achievable through strategic deployment of STAR-RIS, focusing on mitigating inter-cell interference, enhancing signal strength, and extending coverage to cell-edge users. Resource sharing strategies for STAR-RIS elements are explored, optimizing both transmission and reflection functionalities. Analytical frameworks are developed to quantify the benefits of STAR-RIS assisted CoMP-NOMA networks under realistic channel conditions, deriving key performance metrics such as ergodic rates and outage probabilities. Additionally, the research delves into energy-efficient design approaches for CoMP-NOMA networks incorporating RIS, proposing novel RIS configurations and optimization algorithms to achieve a balance between performance and energy consumption. Furthermore, the application of Deep Reinforcement Learning (DRL) techniques for intelligent and adaptive optimization in aerial RIS-assisted CoMP-NOMA networks is explored, aiming to maximize network sum rate while meeting user quality of service requirements. Through a comprehensive investigation of these technologies and their synergistic potential, this thesis contributes valuable insights into the future of wireless communication, paving the way for the development of more efficient, reliable, and sustainable networks capable of meeting the demands of our increasingly connected world. 

**Abstract (ZH)**: 本论文探讨了无线通信前沿技术，通过研究STAR-RIS、CoMP和NOMA三种 transformative 技术的协同集成。受6G发展中对更高数据速率、更佳频谱效率和更广覆盖范围的不断增长需求驱动，本研究调查了这些技术未来无线网络革命化潜力。论文分析了通过战略性部署STAR-RIS实现的性能提升，重点关注干扰抑制、信号增强以及边缘用户的覆盖扩展。探讨了STAR-RIS元素的资源共享策略，优化了传输和反射功能。开发了在实际信道条件下量化STAR-RIS辅助CoMP-NOMA网络效益的分析框架，推导出了关键性能指标，如遍历速率和 outage 概率。此外，研究了结合RIS的CoMP-NOMA网络的节能设计方法，提出了新的RIS配置和优化算法，以实现性能与能耗之间的平衡。同时，探索了在空中RIS辅助CoMP-NOMA网络中应用深度强化学习（DRL）技术进行智能自适应优化的方法，旨在最大化网络总速率并满足用户服务质量要求。通过全面研究这些技术和它们的协同潜力，本论文为未来无线通信提供了宝贵的见解，为开发更高效、可靠和可持续的网络奠定了基础，以满足我们日益互联世界的需求。 

---
# SentenceKV: Efficient LLM Inference via Sentence-Level Semantic KV Caching 

**Title (ZH)**: SentenceKV：通过句子级语义KV缓存实现高效的LLM推理 

**Authors**: Yuxuan Zhu, Ali Falahati, David H. Yang, Mohammad Mohammadi Amiri  

**Link**: [PDF](https://arxiv.org/pdf/2504.00970)  

**Abstract**: Large language models face significant computational and memory challenges when processing long contexts. During inference, efficient management of the key-value (KV) cache, which stores intermediate activations for autoregressive generation, is critical to reducing memory overhead and improving computational efficiency. Traditional token-level efficient KV caching methods overlook semantic information, treating tokens independently without considering their semantic relationships. Meanwhile, existing semantic-preserving KV cache management approaches often suffer from substantial memory usage and high time-to-first-token. To address these limitations, we propose SentenceKV, a novel sentence-level semantic KV caching approach designed to enhance inference efficiency while preserving semantic coherence. During prefilling, SentenceKV groups tokens based on sentence-level semantic similarity, compressing sentence representations into concise semantic vectors stored directly on the GPU, while individual KV pairs are offloaded to CPU. During decoding, SentenceKV generates tokens by selectively retrieving semantically relevant sentence-level KV entries, leveraging the semantic similarity between the prefilling-stage semantic vectors and decoding-stage queries. This ensures efficient and contextually accurate predictions, minimizing the loading of redundant or irrelevant data into GPU memory and significantly reducing memory overhead while maintaining stable inference latency, even for extremely long contexts. Extensive evaluations on benchmarks including PG-19, LongBench, and Needle-In-A-Haystack demonstrate that SentenceKV significantly outperforms state-of-the-art methods in both efficiency and memory usage, without compromising model accuracy. 

**Abstract (ZH)**: SentenceKV：面向长上下文的句级语义键值缓存方法 

---
# HDVIO2.0: Wind and Disturbance Estimation with Hybrid Dynamics VIO 

**Title (ZH)**: HDVIO2.0: 结合混合动力学VIO的风和干扰估计 

**Authors**: Giovanni Cioffi, Leonard Bauersfeld, Davide Scaramuzza  

**Link**: [PDF](https://arxiv.org/pdf/2504.00969)  

**Abstract**: Visual-inertial odometry (VIO) is widely used for state estimation in autonomous micro aerial vehicles using onboard sensors. Current methods improve VIO by incorporating a model of the translational vehicle dynamics, yet their performance degrades when faced with low-accuracy vehicle models or continuous external disturbances, like wind. Additionally, incorporating rotational dynamics in these models is computationally intractable when they are deployed in online applications, e.g., in a closed-loop control system. We present HDVIO2.0, which models full 6-DoF, translational and rotational, vehicle dynamics and tightly incorporates them into a VIO with minimal impact on the runtime. HDVIO2.0 builds upon the previous work, HDVIO, and addresses these challenges through a hybrid dynamics model combining a point-mass vehicle model with a learning-based component, with access to control commands and IMU history, to capture complex aerodynamic effects. The key idea behind modeling the rotational dynamics is to represent them with continuous-time functions. HDVIO2.0 leverages the divergence between the actual motion and the predicted motion from the hybrid dynamics model to estimate external forces as well as the robot state. Our system surpasses the performance of state-of-the-art methods in experiments using public and new drone dynamics datasets, as well as real-world flights in winds up to 25 km/h. Unlike existing approaches, we also show that accurate vehicle dynamics predictions are achievable without precise knowledge of the full vehicle state. 

**Abstract (ZH)**: 基于视觉-惯性里程计学（VIO）的全6自由度车辆动力学建模与实时应用（HDVIO2.0） 

---
# Enabling Efficient Processing of Spiking Neural Networks with On-Chip Learning on Commodity Neuromorphic Processors for Edge AI Systems 

**Title (ZH)**: 在现货神经形态处理器上实现芯片内学习以高效处理脉冲神经网络，应用于边缘AI系统 

**Authors**: Rachmad Vidya Wicaksana Putra, Pasindu Wickramasinghe, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2504.00957)  

**Abstract**: The rising demand for energy-efficient edge AI systems (e.g., mobile agents/robots) has increased the interest in neuromorphic computing, since it offers ultra-low power/energy AI computation through spiking neural network (SNN) algorithms on neuromorphic processors. However, their efficient implementation strategy has not been comprehensively studied, hence limiting SNN deployments for edge AI systems. Toward this, we propose a design methodology to enable efficient SNN processing on commodity neuromorphic processors. To do this, we first study the key characteristics of targeted neuromorphic hardware (e.g., memory and compute budgets), and leverage this information to perform compatibility analysis for network selection. Afterward, we employ a mapping strategy for efficient SNN implementation on the targeted processor. Furthermore, we incorporate an efficient on-chip learning mechanism to update the systems' knowledge for adapting to new input classes and dynamic environments. The experimental results show that the proposed methodology leads the system to achieve low latency of inference (i.e., less than 50ms for image classification, less than 200ms for real-time object detection in video streaming, and less than 1ms in keyword recognition) and low latency of on-chip learning (i.e., less than 2ms for keyword recognition), while incurring less than 250mW of processing power and less than 15mJ of energy consumption across the respective different applications and scenarios. These results show the potential of the proposed methodology in enabling efficient edge AI systems for diverse application use-cases. 

**Abstract (ZH)**: 边缘AI系统中能源效率提升的需求促使了类脑计算的兴趣增加，类脑计算通过神经形态处理器上的脉冲神经网络（SNN）算法提供了超低功耗的AI计算。然而，其高效的实现策略尚未得到全面研究，限制了SNN在边缘AI系统中的部署。为此，我们提出了一种设计方法，以在商用神经形态处理器上实现高效的SNN处理。为此，我们首先研究了目标神经形态硬件的关键特性（如存储和计算预算），并利用这些信息进行网络选择的兼容性分析。随后，我们采用了一种映射策略，以在目标处理器上高效实现SNN。此外，我们引入了一种高效的片内学习机制，以更新系统的知识，使其能够适应新的输入类别和动态环境。实验结果显示，所提出的方法使系统实现了较低的推理延迟（如图像分类少于50毫秒，视频流中实时物体检测少于200毫秒，关键词识别少于1毫秒）和较低的片内学习延迟（如关键词识别少于2毫秒），同时消耗的处理功率少于250毫瓦，能量消耗少于15毫焦，适用于不同的应用和场景。这些结果展示了所提出方法在为多种应用场景提供高效边缘AI系统方面的潜力。 

---
# Unfair Learning: GenAI Exceptionalism and Copyright Law 

**Title (ZH)**: 不公平的学习：GenAI例外主义与版权法 

**Authors**: David Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2504.00955)  

**Abstract**: This paper challenges the argument that generative artificial intelligence (GenAI) is entitled to broad immunity from copyright law for reproducing copyrighted works without authorization due to a fair use defense. It examines fair use legal arguments and eight distinct substantive arguments, contending that every legal and substantive argument favoring fair use for GenAI applies equally, if not more so, to humans. Therefore, granting GenAI exceptional privileges in this domain is legally and logically inconsistent with withholding broad fair use exemptions from individual humans. It would mean no human would need to pay for virtually any copyright work again. The solution is to take a circumspect view of any fair use claim for mass copyright reproduction by any entity and focus on the first principles of whether permitting such exceptionalism for GenAI promotes science and the arts. 

**Abstract (ZH)**: 本文挑战了生成式人工智能（GenAI）在未经授权复制受版权保护的作品时，因公平使用抗辩而享有广泛版权法律豁免权的说法。本文考察了公平使用法律论点和八项具体的实质论点，认为支持GenAI公平使用的每一项法律和实质论点同样适用于个人，甚至更为适用。因此，给予GenAI在这方面享有特权与对个人不予广泛公平使用豁免是法律和逻辑上不一致的。这样意味着人类无需再次为几乎任何版权作品付费。解决方案是谨慎对待任何实体大规模版权复制的公平使用主张，并关注允许这种例外主义是否促进科学和艺术发展。 

---
# IDMR: Towards Instance-Driven Precise Visual Correspondence in Multimodal Retrieval 

**Title (ZH)**: IDMR：迈向实例驱动的精确多模态检索视图对应 

**Authors**: Bangwei Liu, Yicheng Bao, Shaohui Lin, Xuhong Wang, Xin Tan, Yingchun Wang, Yuan Xie, Chaochao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00954)  

**Abstract**: Multimodal retrieval systems are becoming increasingly vital for cutting-edge AI technologies, such as embodied AI and AI-driven digital content industries. However, current multimodal retrieval tasks lack sufficient complexity and demonstrate limited practical application value. It spires us to design Instance-Driven Multimodal Image Retrieval (IDMR), a novel task that requires models to retrieve images containing the same instance as a query image while matching a text-described scenario. Unlike existing retrieval tasks focused on global image similarity or category-level matching, IDMR demands fine-grained instance-level consistency across diverse contexts. To benchmark this capability, we develop IDMR-bench using real-world object tracking and first-person video data. Addressing the scarcity of training data, we propose a cross-domain synthesis method that creates 557K training samples by cropping objects from standard detection datasets. Our Multimodal Large Language Model (MLLM) based retrieval model, trained on 1.2M samples, outperforms state-of-the-art approaches on both traditional benchmarks and our zero-shot IDMR-bench. Experimental results demonstrate previous models' limitations in instance-aware retrieval and highlight the potential of MLLM for advanced retrieval applications. The whole training dataset, codes and models, with wide ranges of sizes, are available at this https URL. 

**Abstract (ZH)**: 多模态检索系统对于 embodied AI 和 AI 驱动的数字内容产业等前沿 AI 技术变得越来越重要。然而，当前的多模态检索任务缺乏足够的复杂性，展示出有限的实际应用价值。因此，我们设计了基于实例的多模态图像检索 (IDMR)，这是一个需要模型检索包含查询图像中相同实例的同时匹配文本描述场景的新型任务。与现有专注于全局图像相似度或类别级别匹配的检索任务不同，IDMR 要求在多种情境下保持精细的实例级一致性。为评估这一能力，我们使用真实世界的对象跟踪和第一人称视频数据开发了 IDMR-bench。为应对训练数据稀缺的问题，我们提出了一种跨领域合成方法，通过从标准检测数据集中裁剪对象生成了 55.7 万训练样本。基于 Multimodal 大语言模型 (MLLM) 的检索模型，该模型在 120 万样本上训练，不仅在传统基准测试上超过了现有方法，还在我们提出的零样本 IDMR-bench 上表现出色。实验结果表明了先前模型在实例感知检索方面的局限性，并强调了 MLLM 在高级检索应用中的潜力。整个训练数据集、代码和模型，具有广泛的大小范围，可在以下网址获得。 

---
# Personalized Federated Training of Diffusion Models with Privacy Guarantees 

**Title (ZH)**: 带有隐私保证的个性化联邦扩散模型训练 

**Authors**: Kumar Kshitij Patel, Weitong Zhang, Lingxiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00952)  

**Abstract**: The scarcity of accessible, compliant, and ethically sourced data presents a considerable challenge to the adoption of artificial intelligence (AI) in sensitive fields like healthcare, finance, and biomedical research. Furthermore, access to unrestricted public datasets is increasingly constrained due to rising concerns over privacy, copyright, and competition. Synthetic data has emerged as a promising alternative, and diffusion models -- a cutting-edge generative AI technology -- provide an effective solution for generating high-quality and diverse synthetic data. In this paper, we introduce a novel federated learning framework for training diffusion models on decentralized private datasets. Our framework leverages personalization and the inherent noise in the forward diffusion process to produce high-quality samples while ensuring robust differential privacy guarantees. Our experiments show that our framework outperforms non-collaborative training methods, particularly in settings with high data heterogeneity, and effectively reduces biases and imbalances in synthetic data, resulting in fairer downstream models. 

**Abstract (ZH)**: 可访问、合规且伦理来源数据的稀缺性对医疗、金融和生物医学研究等领域中人工智能（AI）的应用构成了重大挑战。此外，由于对隐私、版权和竞争的担忧日益增加，不可限制的公共数据集的访问也越来越受到限制。合成数据作为一种有前途的替代方案已经出现，而基于最新生成AI技术的扩散模型为生成高质量和多样化的合成数据提供了一个有效的解决方案。在本文中，我们介绍了一种新的联邦学习框架，用于在分布式私有数据集上训练扩散模型。我们的框架利用个性化和正向扩散过程固有的噪声来生成高质量样本，同时确保强大的差分隐私保证。实验结果表明，我们的框架在数据异质性高的情况下优于非协作训练方法，有效减少了合成数据中的偏差和不平衡，从而提高了下游模型的公平性。 

---
# QSViT: A Methodology for Quantizing Spiking Vision Transformers 

**Title (ZH)**: QSViT：量化脉冲视觉变换器的方法ology 

**Authors**: Rachmad Vidya Wicaksana Putra, Saad Iftikhar, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2504.00948)  

**Abstract**: Vision Transformer (ViT)-based models have shown state-of-the-art performance (e.g., accuracy) in vision-based AI tasks. However, realizing their capability in resource-constrained embedded AI systems is challenging due to their inherent large memory footprints and complex computations, thereby incurring high power/energy consumption. Recently, Spiking Vision Transformer (SViT)-based models have emerged as alternate low-power ViT networks. However, their large memory footprints still hinder their applicability for resource-constrained embedded AI systems. Therefore, there is a need for a methodology to compress SViT models without degrading the accuracy significantly. To address this, we propose QSViT, a novel design methodology to compress the SViT models through a systematic quantization strategy across different network layers. To do this, our QSViT employs several key steps: (1) investigating the impact of different precision levels in different network layers, (2) identifying the appropriate base quantization settings for guiding bit precision reduction, (3) performing a guided quantization strategy based on the base settings to select the appropriate quantization setting, and (4) developing an efficient quantized network based on the selected quantization setting. The experimental results demonstrate that, our QSViT methodology achieves 22.75% memory saving and 21.33% power saving, while also maintaining high accuracy within 2.1% from that of the original non-quantized SViT model on the ImageNet dataset. These results highlight the potential of QSViT methodology to pave the way toward the efficient SViT deployments on resource-constrained embedded AI systems. 

**Abstract (ZH)**: 基于Spiking Vision Transformer (SViT)的模型压缩方法：QSViT在资源受限嵌入式AI系统的高效部署 

---
# Graph Classification and Radiomics Signature for Identification of Tuberculous Meningitis 

**Title (ZH)**: 基于图分类和放射omics特征标识结核性脑膜炎 

**Authors**: Snigdha Agarwal, Ganaraja V H, Neelam Sinha, Abhilasha Indoria, Netravathi M, Jitender Saini  

**Link**: [PDF](https://arxiv.org/pdf/2504.00943)  

**Abstract**: Introduction: Tuberculous meningitis (TBM) is a serious brain infection caused by Mycobacterium tuberculosis, characterized by inflammation of the meninges covering the brain and spinal cord. Diagnosis often requires invasive lumbar puncture (LP) and cerebrospinal fluid (CSF) analysis. Objectives: This study aims to classify TBM patients using T1-weighted (T1w) non-contrast Magnetic Resonance Imaging (MRI) scans. We hypothesize that specific brain regions, such as the interpeduncular cisterns, bone, and corpus callosum, contain visual markers that can non-invasively distinguish TBM patients from healthy controls. We propose a novel Pixel-array Graphs Classifier (PAG-Classifier) that leverages spatial relationships between neighbouring 3D pixels in a graph-based framework to extract significant features through eigen decomposition. These features are then used to train machine learning classifiers for effective patient classification. We validate our approach using a radiomics-based methodology, classifying TBM patients based on relevant radiomics features. Results: We utilized an internal dataset consisting of 52 scans, 32 from confirmed TBM patients based on mycobacteria detection in CSF, and 20 from healthy individuals. We achieved a 5-fold cross-validated average F1 score of 85.71% for cistern regions with our PAG-Classifier and 92.85% with the radiomics features classifier, surpassing current state-of-the-art benchmarks by 15% and 22%, respectively. However, bone and corpus callosum regions showed poor classification effectiveness, with average F1 scores below 50%. Conclusion: Our study suggests that algorithms like the PAG-Classifier serve as effective tools for non-invasive TBM analysis, particularly by targeting the interpeduncular cistern. Findings indicate that the bone and corpus callosum regions lack distinctive patterns for differentiation. 

**Abstract (ZH)**: Tuberculous 脑膜炎患者基于 T1 加权非对比磁共振成像的分类研究：像素阵列图形分类器的探索 

---
# Role and Use of Race in AI/ML Models Related to Health 

**Title (ZH)**: AI/ML模型在健康领域中种族的角色与应用 

**Authors**: Martin C. Were, Ang Li, Bradley A. Malin, Zhijun Yin, Joseph R. Coco, Benjamin X. Collins, Ellen Wright Clayton, Laurie L. Novak, Rachele Hendricks-Sturrup, Abiodun Oluyomi, Shilo Anders, Chao Yan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00899)  

**Abstract**: The role and use of race within health-related artificial intelligence and machine learning (AI/ML) models has sparked increasing attention and controversy. Despite the complexity and breadth of related issues, a robust and holistic framework to guide stakeholders in their examination and resolution remains lacking. This perspective provides a broad-based, systematic, and cross-cutting landscape analysis of race-related challenges, structured around the AI/ML lifecycle and framed through "points to consider" to support inquiry and decision-making. 

**Abstract (ZH)**: 健康相关人工智能和机器学习模型中种族问题的作用与应用引发了广泛关注和争议。尽管相关问题复杂且广泛，但仍缺乏一个全面和综合的框架来指导相关利益方的审查和解决。本文提供了关于种族相关挑战的广泛、系统和综合的景观分析，围绕人工智能和机器学习生命周期展开，并通过“需考虑的要点”来支持探索和决策。 

---
# Spectral Architecture Search for Neural Networks 

**Title (ZH)**: 神经网络的光谱架构搜索 

**Authors**: Gianluca Peri, Lorenzo Giambagli, Lorenzo Chicchi, Duccio Fanelli  

**Link**: [PDF](https://arxiv.org/pdf/2504.00885)  

**Abstract**: Architecture design and optimization are challenging problems in the field of artificial neural networks. Working in this context, we here present SPARCS (SPectral ARchiteCture Search), a novel architecture search protocol which exploits the spectral attributes of the inter-layer transfer matrices. SPARCS allows one to explore the space of possible architectures by spanning continuous and differentiable manifolds, thus enabling for gradient-based optimization algorithms to be eventually employed. With reference to simple benchmark models, we show that the newly proposed method yields a self-emerging architecture with a minimal degree of expressivity to handle the task under investigation and with a reduced parameter count as compared to other viable alternatives. 

**Abstract (ZH)**: 基于人工神经网络的架构设计与优化是具有挑战性的问题。在此背景下，我们提出SPARCS（SPectral ARchiteCture Search），这是一种新颖的架构搜索协议，利用层间传输矩阵的谱属性。SPARCS通过扩展连续和可微流形来探索可能架构的空间，从而使得基于梯度的优化算法得以应用。参考简单的基准模型，我们展示了新提出的 方法生成了一个自涌现架构，该架构具有处理所研究任务所需的最小表达能力，并且参数数量较少，与其他可行的替代方案相比。 

---
# Improved Visual-Spatial Reasoning via R1-Zero-Like Training 

**Title (ZH)**: 通过R1-Zero-like训练提高视觉-空间推理能力 

**Authors**: Zhenyi Liao, Qingsong Xie, Yanhao Zhang, Zijian Kong, Haonan Lu, Zhenyu Yang, Zhijie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2504.00883)  

**Abstract**: Increasing attention has been placed on improving the reasoning capacities of multi-modal large language models (MLLMs). As the cornerstone for AI agents that function in the physical realm, video-based visual-spatial intelligence (VSI) emerges as one of the most pivotal reasoning capabilities of MLLMs. This work conducts a first, in-depth study on improving the visual-spatial reasoning of MLLMs via R1-Zero-like training. Technically, we first identify that the visual-spatial reasoning capacities of small- to medium-sized Qwen2-VL models cannot be activated via Chain of Thought (CoT) prompts. We then incorporate GRPO training for improved visual-spatial reasoning, using the carefully curated VSI-100k dataset, following DeepSeek-R1-Zero. During the investigation, we identify the necessity to keep the KL penalty (even with a small value) in GRPO. With just 120 GPU hours, our vsGRPO-2B model, fine-tuned from Qwen2-VL-2B, can outperform the base model by 12.1% and surpass GPT-4o. Moreover, our vsGRPO-7B model, fine-tuned from Qwen2-VL-7B, achieves performance comparable to that of the best open-source model LLaVA-NeXT-Video-72B. Additionally, we compare vsGRPO to supervised fine-tuning and direct preference optimization baselines and observe strong performance superiority. The code and dataset will be available soon. 

**Abstract (ZH)**: 提升多模态大型语言模型的视觉-空间推理能力：基于R1-Zero-like训练的方法 

---
# CrackSQL: A Hybrid SQL Dialect Translation System Powered by Large Language Models 

**Title (ZH)**: CrackSQL: 由大规模语言模型驱动的混合SQL方言翻译系统 

**Authors**: Wei Zhou, Yuyang Gao, Xuanhe Zhou, Guoliang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.00882)  

**Abstract**: Dialect translation plays a key role in enabling seamless interaction across heterogeneous database systems. However, translating SQL queries between different dialects (e.g., from PostgreSQL to MySQL) remains a challenging task due to syntactic discrepancies and subtle semantic variations. Existing approaches including manual rewriting, rule-based systems, and large language model (LLM)-based techniques often involve high maintenance effort (e.g., crafting custom translation rules) or produce unreliable results (e.g., LLM generates non-existent functions), especially when handling complex queries. In this demonstration, we present CrackSQL, the first hybrid SQL dialect translation system that combines rule and LLM-based methods to overcome these limitations. CrackSQL leverages the adaptability of LLMs to minimize manual intervention, while enhancing translation accuracy by segmenting lengthy complex SQL via functionality-based query processing. To further improve robustness, it incorporates a novel cross-dialect syntax embedding model for precise syntax alignment, as well as an adaptive local-to-global translation strategy that effectively resolves interdependent query operations. CrackSQL supports three translation modes and offers multiple deployment and access options including a web console interface, a PyPI package, and a command-line prompt, facilitating adoption across a variety of real-world use cases 

**Abstract (ZH)**: 方言翻译在实现异构数据库系统之间无缝交互中起着关键作用。然而，不同方言之间的SQL查询翻译（例如，从PostgreSQL到MySQL）由于句法差异和微妙的语义变化仍是一项具有挑战性的任务。现有的方法包括手动重写、基于规则的系统和基于大型语言模型（LLM）的技术，往往需要高维护 effort（例如，制定自定义翻译规则）或产生不可靠的结果（例如，LLM生成不存在的函数），特别是在处理复杂查询时。在这项演示中，我们介绍了CrackSQL，这是一种结合规则和LLM方法的首个混合SQL方言翻译系统，以克服这些限制。CrackSQL利用LLM的适应性来减少人工干预，通过基于功能的查询处理将长而复杂的SQL拆分，以提高翻译准确性。为了进一步提高鲁棒性，它引入了一种新的跨方言句法嵌入模型进行精确的语法对齐，以及一种有效的解决相互依赖查询操作的自适应局部到全局翻译策略。CrackSQL支持三种翻译模式，并提供多种部署和访问选项，包括Web控制台接口、PyPI包和命令行提示，便于在各种实际应用案例中采用。 

---
# m1: Unleash the Potential of Test-Time Scaling for Medical Reasoning with Large Language Models 

**Title (ZH)**: M1: 在大规模语言模型中释放测试时缩放的潜力以促进医疗推理 

**Authors**: Xiaoke Huang, Juncheng Wu, Hui Liu, Xianfeng Tang, Yuyin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.00869)  

**Abstract**: Test-time scaling has emerged as a powerful technique for enhancing the reasoning capabilities of large language models. However, its effectiveness in medical reasoning remains uncertain, as the medical domain fundamentally differs from mathematical tasks in terms of knowledge representation and decision-making processes. In this paper, we provide the first comprehensive investigation of test-time scaling for medical reasoning and present m1, a simple yet effective approach that increases a model's medical reasoning capability at inference. Our evaluation across diverse medical tasks demonstrates that test-time scaling consistently enhances medical reasoning, enabling lightweight fine-tuned models under 10B parameters to establish new state-of-the-art performance, while our 32B model rivals previous 70B-scale medical LLMs. However, we identify an optimal reasoning token budget of approximately 4K, beyond which performance may degrade due to overthinking. Budget forcing, which extends test-time computation through iterative prompts, helps models double-check answers but does not necessarily improve the overall medical QA performance and, in some cases, even introduces errors into previously correct responses. Our case-by-case analysis identifies insufficient medical knowledge as a key bottleneck that prevents further performance gains through test-time scaling. We find that increasing data scale, improving data quality, and expanding model capacity consistently enhance medical knowledge grounding, enabling continued performance improvements, particularly on challenging medical benchmarks where smaller models reach saturation. These findings underscore fundamental differences between medical and mathematical reasoning in LLMs, highlighting that enriched medical knowledge, other than increased reasoning depth alone, is essential for realizing the benefits of test-time scaling. 

**Abstract (ZH)**: 测试时缩放已成为增强大规模语言模型推理能力的有力技术，但在医疗推理领域的有效性尚不确定，因为医疗领域在知识表示和决策过程中与数学任务存在根本性差异。本文首次全面探讨了医疗推理中的测试时缩放，并提出了m1，这是一种简单而有效的方法，可以提高模型在推理时的医疗推理能力。我们的跨多种医疗任务评估表明，测试时缩放能够一致地增强医疗推理能力，使轻量级细调模型在少于10B参数的情况下取得新的最佳性能，而我们的32B模型能与之前的70B规模的医疗LLM相媲美。然而，我们发现了一个大约4K的最优推理令牌预算，超过这个预算可能会因为过度推理而导致性能下降。预算强制策略通过迭代提示扩展测试时计算，有助于模型复查答案，但并不一定能改善整体医疗问答性能，在某些情况下甚至会引入错误到先前正确的回答中。我们的逐案分析发现，缺乏医疗知识是阻碍通过测试时缩放进一步性能提升的关键瓶颈。我们发现，增加数据规模、提高数据质量以及扩大模型容量能够一致地增强医疗知识基础，使模型在难以处理的医疗基准上持续取得性能提升。这些发现强调了LLM中的医疗与数学推理之间根本差异，并突出了除了增加推理深度之外，丰富医疗知识对于实现测试时缩放效益的必要性。 

---
# Investigating the Capabilities and Limitations of Machine Learning for Identifying Bias in English Language Data with Information and Heritage Professionals 

**Title (ZH)**: 探究机器学习在识别英语语言数据中的偏见方面的能力和局限性——以信息和遗产专业人员为例 

**Authors**: Lucy Havens, Benjamin Bach, Melissa Terras, Beatrice Alex  

**Link**: [PDF](https://arxiv.org/pdf/2504.00860)  

**Abstract**: Despite numerous efforts to mitigate their biases, ML systems continue to harm already-marginalized people. While predominant ML approaches assume bias can be removed and fair models can be created, we show that these are not always possible, nor desirable, goals. We reframe the problem of ML bias by creating models to identify biased language, drawing attention to a dataset's biases rather than trying to remove them. Then, through a workshop, we evaluated the models for a specific use case: workflows of information and heritage professionals. Our findings demonstrate the limitations of ML for identifying bias due to its contextual nature, the way in which approaches to mitigating it can simultaneously privilege and oppress different communities, and its inevitability. We demonstrate the need to expand ML approaches to bias and fairness, providing a mixed-methods approach to investigating the feasibility of removing bias or achieving fairness in a given ML use case. 

**Abstract (ZH)**: 尽管付出了大量努力来减轻其偏见，机器学习系统仍继续伤害已处于不利地位的人群。尽管主流的机器学习方法假设可以消除偏见并创建公平模型，但我们表明，这并非总是可行或可取的目标。我们通过创建模型来识别有偏见的语言，重新定义机器学习偏见问题，从而将注意力集中在数据集的偏见上，而不是试图消除它们。随后，通过研讨会，我们评估了这些模型在具体用例中的适用性：信息和遗产专业人士的工作流程。研究结果证明了因上下文关系而导致的机器学习在识别偏见方面的局限性，以及缓解偏见的方法可能会同时特权和压迫不同的社群，并且偏见是不可避免的。我们展示了需要扩展机器学习在偏见和公平性方面的研究，提供混合方法来探究在特定机器学习用例中去除偏见或实现公平性的可行性。 

---
# Exploring Personalized Federated Learning Architectures for Violence Detection in Surveillance Videos 

**Title (ZH)**: 探索针对监控视频中暴力检测的个性化联邦学习架构 

**Authors**: Mohammad Kassir, Siba Haidar, Antoun Yaacoub  

**Link**: [PDF](https://arxiv.org/pdf/2504.00857)  

**Abstract**: The challenge of detecting violent incidents in urban surveillance systems is compounded by the voluminous and diverse nature of video data. This paper presents a targeted approach using Personalized Federated Learning (PFL) to address these issues, specifically employing the Federated Learning with Personalization Layers method within the Flower framework. Our methodology adapts learning models to the unique data characteristics of each surveillance node, effectively managing the heterogeneous and non-IID nature of surveillance video data. Through rigorous experiments conducted on balanced and imbalanced datasets, our PFL models demonstrated enhanced accuracy and efficiency, achieving up to 99.3% accuracy. This study underscores the potential of PFL to significantly improve the scalability and effectiveness of surveillance systems, offering a robust, privacy-preserving solution for violence detection in complex urban environments. 

**Abstract (ZH)**: 基于个性化联邦学习的城市 surveillance 系统中暴力事件检测挑战及其解决方法 

---
# ReaLitE: Enrichment of Relation Embeddings in Knowledge Graphs using Numeric Literals 

**Title (ZH)**: ReaLitE：在知识图中利用数值_LITERAL_丰富关系嵌入 

**Authors**: Antonis Klironomos, Baifan Zhou, Zhuoxun Zheng, Gad-Elrab Mohamed, Heiko Paulheim, Evgeny Kharlamov  

**Link**: [PDF](https://arxiv.org/pdf/2504.00852)  

**Abstract**: Most knowledge graph embedding (KGE) methods tailored for link prediction focus on the entities and relations in the graph, giving little attention to other literal values, which might encode important information. Therefore, some literal-aware KGE models attempt to either integrate numerical values into the embeddings of the entities or convert these numerics into entities during preprocessing, leading to information loss. Other methods concerned with creating relation-specific numerical features assume completeness of numerical data, which does not apply to real-world graphs. In this work, we propose ReaLitE, a novel relation-centric KGE model that dynamically aggregates and merges entities' numerical attributes with the embeddings of the connecting relations. ReaLitE is designed to complement existing conventional KGE methods while supporting multiple variations for numerical aggregations, including a learnable method.
We comprehensively evaluated the proposed relation-centric embedding using several benchmarks for link prediction and node classification tasks. The results showed the superiority of ReaLitE over the state of the art in both tasks. 

**Abstract (ZH)**: 关系中心的ReaLitE：一种动态聚合和融合实体数值属性的知识图嵌入模型 

---
# Global Intervention and Distillation for Federated Out-of-Distribution Generalization 

**Title (ZH)**: 全球干预与蒸馏在联邦领域外泛化的应用 

**Authors**: Zhuang Qi, Runhui Zhang, Lei Meng, Wei Wu, Yachong Zhang, Xiangxu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2504.00850)  

**Abstract**: Attribute skew in federated learning leads local models to focus on learning non-causal associations, guiding them towards inconsistent optimization directions, which inevitably results in performance degradation and unstable convergence. Existing methods typically leverage data augmentation to enhance sample diversity or employ knowledge distillation to learn invariant representations. However, the instability in the quality of generated data and the lack of domain information limit their performance on unseen samples. To address these issues, this paper presents a global intervention and distillation method, termed FedGID, which utilizes diverse attribute features for backdoor adjustment to break the spurious association between background and label. It includes two main modules, where the global intervention module adaptively decouples objects and backgrounds in images, injects background information into random samples to intervene in the sample distribution, which links backgrounds to all categories to prevent the model from treating background-label associations as causal. The global distillation module leverages a unified knowledge base to guide the representation learning of client models, preventing local models from overfitting to client-specific attributes. Experimental results on three datasets demonstrate that FedGID enhances the model's ability to focus on the main subjects in unseen data and outperforms existing methods in collaborative modeling. 

**Abstract (ZH)**: 联邦学习中属性偏差导致局部模型聚焦于学习非因果关联，引导它们朝着不一致的优化方向发展，从而不可避免地导致性能下降和收敛不稳定。现有方法通常依赖数据增强以增强样本多样性或采用知识蒸馏以学习不变表示。然而，生成数据质量的不稳定性和领域信息的缺乏限制了其在未见过样本上的性能。为解决这些问题，本文提出了一种全球干预和蒸馏方法，命名为FedGID，利用多样化的属性特征进行后门调整以打破背景与标签之间的虚假关联。该方法包括两个主要模块，其中全球干预模块自适应地将图像中的对象和背景分离，向随机样本注入背景信息以干预样本分布，将背景与所有类别联系起来，防止模型将背景-标签关联视为因果关系。全球蒸馏模块利用统一的知识库指导客户端模型的表示学习，防止局部模型过度拟合于客户端特定的属性。在三个数据集上的实验结果表明，FedGID提升了模型在未见过数据中聚焦主要主体的能力，并在联合建模中优于现有方法。 

---
# Context-Aware Human Behavior Prediction Using Multimodal Large Language Models: Challenges and Insights 

**Title (ZH)**: 基于多模态大语言模型的.context-aware人类行为预测：挑战与见解 

**Authors**: Yuchen Liu, Lino Lerch, Luigi Palmieri, Andrey Rudenko, Sebastian Koch, Timo Ropinski, Marco Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2504.00839)  

**Abstract**: Predicting human behavior in shared environments is crucial for safe and efficient human-robot interaction. Traditional data-driven methods to that end are pre-trained on domain-specific datasets, activity types, and prediction horizons. In contrast, the recent breakthroughs in Large Language Models (LLMs) promise open-ended cross-domain generalization to describe various human activities and make predictions in any context. In particular, Multimodal LLMs (MLLMs) are able to integrate information from various sources, achieving more contextual awareness and improved scene understanding. The difficulty in applying general-purpose MLLMs directly for prediction stems from their limited capacity for processing large input sequences, sensitivity to prompt design, and expensive fine-tuning. In this paper, we present a systematic analysis of applying pre-trained MLLMs for context-aware human behavior prediction. To this end, we introduce a modular multimodal human activity prediction framework that allows us to benchmark various MLLMs, input variations, In-Context Learning (ICL), and autoregressive techniques. Our evaluation indicates that the best-performing framework configuration is able to reach 92.8% semantic similarity and 66.1% exact label accuracy in predicting human behaviors in the target frame. 

**Abstract (ZH)**: 共享环境中的人类行为预测对于安全高效的机器人交互至关重要。传统的数据驱动方法依赖于特定领域的数据集、活动类型和预测时间窗口进行预训练。相比之下，大型语言模型（LLMs）的近期突破允诺了跨领域的开放性泛化能力，以描述各种人类活动并在任何上下文中进行预测。特别是多模态LLMs能够从多种来源整合信息，从而实现更强的上下文意识和更优的场景理解。直接将通用的多模态LLMs应用于预测的一大困难在于它们处理大型输入序列的能力有限、对提示设计高度敏感且微调成本高昂。在本文中，我们提出了一个系统化的分析，探讨预训练的多模态LLMs在上下文感知的人类行为预测中的应用。为此，我们引入了一个模块化的多模态人类活动预测框架，使我们能够对各种LLMs、输入变化、增量式学习（ICL）和自回归技术进行基准测试。我们的评估表明，最佳框架配置能够在目标帧中达到92.8%的语义相似度和66.1%的精确标签准确性。 

---
# A Survey on Music Generation from Single-Modal, Cross-Modal, and Multi-Modal Perspectives: Data, Methods, and Challenges 

**Title (ZH)**: 单模态、跨模态和多模态视角下的音乐生成综述：数据、方法与挑战 

**Authors**: Shuyu Li, Shulei Ji, Zihao Wang, Songruoyao Wu, Jiaxing Yu, Kejun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00837)  

**Abstract**: Multi-modal music generation, using multiple modalities like images, video, and text alongside musical scores and audio as guidance, is an emerging research area with broad applications. This paper reviews this field, categorizing music generation systems from the perspective of modalities. It covers modality representation, multi-modal data alignment, and their utilization to guide music generation. We also discuss current datasets and evaluation methods. Key challenges in this area include effective multi-modal integration, large-scale comprehensive datasets, and systematic evaluation methods. Finally, we provide an outlook on future research directions focusing on multi-modal fusion, alignment, data, and evaluation. 

**Abstract (ZH)**: 多模态音乐生成：利用图像、视频、文本等多种模态与音乐谱和音频作为指导的多模态音乐生成是一个新兴的研究领域，具有广泛的應用。本文从模态的角度回顾该领域，涵盖模态表示、多模态数据对齐及其在音乐生成中的应用，并讨论了当前的数据集和评估方法。该领域面临的挑战包括有效的多模态集成、大规模综合数据集和系统的评估方法。最后，本文展望了未来研究方向，集中在多模态融合、对齐、数据和评估方面。 

---
# Conditional Temporal Neural Processes with Covariance Loss 

**Title (ZH)**: 条件时序神经过程及其协方差损失 

**Authors**: Boseon Yoo, Jiwoo Lee, Janghoon Ju, Seijun Chung, Soyeon Kim, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00794)  

**Abstract**: We introduce a novel loss function, Covariance Loss, which is conceptually equivalent to conditional neural processes and has a form of regularization so that is applicable to many kinds of neural networks. With the proposed loss, mappings from input variables to target variables are highly affected by dependencies of target variables as well as mean activation and mean dependencies of input and target variables. This nature enables the resulting neural networks to become more robust to noisy observations and recapture missing dependencies from prior information. In order to show the validity of the proposed loss, we conduct extensive sets of experiments on real-world datasets with state-of-the-art models and discuss the benefits and drawbacks of the proposed Covariance Loss. 

**Abstract (ZH)**: 我们介绍了一种新的损失函数——协方差损失，该损失函数在概念上等价于条件神经过程，并且具有正则化的形式，使其适用于多种类型的神经网络。这种损失函数使得输入变量到目标变量的映射不仅受到目标变量依赖性的影响，还受到输入和目标变量的均值激活及均值依赖性的影响。这种特性使得得到的神经网络能够更好地应对噪声观测，并从先验信息中还原缺失的依赖性。为了证明所提出损失函数的有效性，我们在使用最新模型的现实世界数据集上进行了广泛的实验，并讨论了所提出协方差损失的优点和缺点。 

---
# Digitally Supported Analysis of Spontaneous Speech (DigiSpon): Benchmarking NLP-Supported Language Sample Analysis of Swiss Children's Speech 

**Title (ZH)**: 数字化支持的自发口语分析 (DigiSpon): 瑞士儿童口语语言样本分析的NLP Benchmarking 

**Authors**: Anja Ryser, Yingqiang Gao, Sarah Ebling  

**Link**: [PDF](https://arxiv.org/pdf/2504.00780)  

**Abstract**: Language sample analysis (LSA) is a process that complements standardized psychometric tests for diagnosing, for example, developmental language disorder (DLD) in children. However, its labor-intensive nature has limited its use in speech-language pathology practice. We introduce an approach that leverages natural language processing (NLP) methods not based on commercial large language models (LLMs) applied to transcribed speech data from 119 children in the German speaking part of Switzerland with typical and atypical language development. The study aims to identify optimal practices that support speech-language pathologists in diagnosing DLD more efficiently within a human-in-the-loop framework, without relying on potentially unethical implementations that leverage commercial LLMs. Preliminary findings underscore the potential of integrating locally deployed NLP methods into the process of semi-automatic LSA. 

**Abstract (ZH)**: 基于自然语言处理的方法在瑞士德语区儿童典型和非典型语言发展数据中的语言样本分析 

---
# LLMs4SchemaDiscovery: A Human-in-the-Loop Workflow for Scientific Schema Mining with Large Language Models 

**Title (ZH)**: LLMs4SchemaDiscovery：大型语言模型辅助的科学模式发现人类在环工作流 

**Authors**: Sameer Sadruddin, Jennifer D'Souza, Eleni Poupaki, Alex Watkins, Hamed Babaei Giglou, Anisa Rula, Bora Karasulu, Sören Auer, Adrie Mackus, Erwin Kessels  

**Link**: [PDF](https://arxiv.org/pdf/2504.00752)  

**Abstract**: Extracting structured information from unstructured text is crucial for modeling real-world processes, but traditional schema mining relies on semi-structured data, limiting scalability. This paper introduces schema-miner, a novel tool that combines large language models with human feedback to automate and refine schema extraction. Through an iterative workflow, it organizes properties from text, incorporates expert input, and integrates domain-specific ontologies for semantic depth. Applied to materials science--specifically atomic layer deposition--schema-miner demonstrates that expert-guided LLMs generate semantically rich schemas suitable for diverse real-world applications. 

**Abstract (ZH)**: 从无结构文本中提取结构化信息对于建模现实世界过程至关重要，但传统的模式挖掘依赖于半结构化数据，限制了其可扩展性。本文介绍了schema-miner，这是一种结合大型语言模型和人类反馈来自动化和精炼模式提取的新工具。通过迭代工作流，它组织文本中的属性，整合专家输入，并结合领域特定本体以实现语义深度。应用于材料科学，特别是原子层沉积，schema-miner 显示出在专家指导下，大型语言模型生成适合多种实际应用场景的语义丰富的模式。 

---
# Advancements in Multimodal Differential Evolution: A Comprehensive Review and Future Perspectives 

**Title (ZH)**: 多模态差分进化算法的发展：综述与未来展望 

**Authors**: Dikshit Chauhan, Shivani, Donghwi Jung, Anupam Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2504.00717)  

**Abstract**: Multi-modal optimization involves identifying multiple global and local optima of a function, offering valuable insights into diverse optimal solutions within the search space. Evolutionary algorithms (EAs) excel at finding multiple solutions in a single run, providing a distinct advantage over classical optimization techniques that often require multiple restarts without guarantee of obtaining diverse solutions. Among these EAs, differential evolution (DE) stands out as a powerful and versatile optimizer for continuous parameter spaces. DE has shown significant success in multi-modal optimization by utilizing its population-based search to promote the formation of multiple stable subpopulations, each targeting different optima. Recent advancements in DE for multi-modal optimization have focused on niching methods, parameter adaptation, hybridization with other algorithms including machine learning, and applications across various domains. Given these developments, it is an opportune moment to present a critical review of the latest literature and identify key future research directions. This paper offers a comprehensive overview of recent DE advancements in multimodal optimization, including methods for handling multiple optima, hybridization with EAs, and machine learning, and highlights a range of real-world applications. Additionally, the paper outlines a set of compelling open problems and future research issues from multiple perspectives 

**Abstract (ZH)**: 多模态优化涉及识别函数的多个全局和局部最优解，为搜索空间内的多种最优解提供有价值的见解。进化算法（EAs）能够在单次运行中找到多个解，这在获得多样解方面远胜于需要多次重启且无法保证得到多样解的古典优化技术。在这些EAs中，差分进化（DE）因其在连续参数空间中作为强大且多功能优化器的突出表现而脱颖而出。DE通过基于群体的搜索机制促进了多个稳定子群体的形成，每个子群体都针对不同的最优解。近年来，DE在多模态优化领域的进步集中在分群方法、参数自适应以及与其他算法（包括机器学习）的结合，以及在各个领域的应用。鉴于这些发展，对最新文献进行批判性回顾并确定关键的未来研究方向恰逢其时。本文提供了最近DE在多模态优化方面的综合概述，包括处理多个最优解的方法、与其他进化算法的结合以及与机器学习的结合，并强调了各种实际应用。此外，本文还从多个视角列出了若干引人入胜的开放问题和未来研究问题。 

---
# Science Autonomy using Machine Learning for Astrobiology 

**Title (ZH)**: 基于机器学习的天体生物学自主科学发现 

**Authors**: Victoria Da Poian, Bethany Theiling, Eric Lyness, David Burtt, Abigail R. Azari, Joey Pasterski, Luoth Chou, Melissa Trainer, Ryan Danell, Desmond Kaplan, Xiang Li, Lily Clough, Brett McKinney, Lukas Mandrake, Bill Diamond, Caroline Freissinet  

**Link**: [PDF](https://arxiv.org/pdf/2504.00709)  

**Abstract**: In recent decades, artificial intelligence (AI) including machine learning (ML) have become vital for space missions enabling rapid data processing, advanced pattern recognition, and enhanced insight extraction. These tools are especially valuable in astrobiology applications, where models must distinguish biotic patterns from complex abiotic backgrounds. Advancing the integration of autonomy through AI and ML into space missions is a complex challenge, and we believe that by focusing on key areas, we can make significant progress and offer practical recommendations for tackling these obstacles. 

**Abstract (ZH)**: 近年来，包括机器学习在内的人工智能在航天任务中变得至关重要，能够实现快速数据处理、高级模式识别和增强的信息提取。这些工具在 astrobiology 应用中尤为重要，因为模型必须在复杂的非生物背景下区分生物模式。通过人工智能和机器学习增强航天任务中的自主集成是一项复杂挑战，我们相信通过聚焦关键领域，可以实现显著进展并提出应对这些挑战的实际建议。 

---
# Energy Weighted Learning Progress Guided Interleaved Multi-Task Learning 

**Title (ZH)**: 能量加权学习进步引导的交错多任务学习 

**Authors**: Hanne Say, Suzan Ece Ada, Emre Ugur, Erhan Oztop  

**Link**: [PDF](https://arxiv.org/pdf/2504.00707)  

**Abstract**: Humans can continuously acquire new skills and knowledge by exploiting existing ones for improved learning, without forgetting them. Similarly, 'continual learning' in machine learning aims to learn new information while preserving the previously acquired knowledge. Existing research often overlooks the nature of human learning, where tasks are interleaved due to human choice or environmental constraints. So, almost never do humans master one task before switching to the next. To investigate to what extent human-like learning can benefit the learner, we propose a method that interleaves tasks based on their 'learning progress' and energy consumption. From a machine learning perspective, our approach can be seen as a multi-task learning system that balances learning performance with energy constraints while mimicking ecologically realistic human task learning. To assess the validity of our approach, we consider a robot learning setting in simulation, where the robot learns the effect of its actions in different contexts. The conducted experiments show that our proposed method achieves better performance than sequential task learning and reduces energy consumption for learning the tasks. 

**Abstract (ZH)**: 人类可以通过利用现有技能和知识来不断获取新技能和知识，从而提高学习效果，而不至于遗忘之前的知识。类似地，机器学习中的“持续学习”旨在学习新信息的同时保留之前获得的知识。现有研究往往忽视了人类学习的特性，由于人类选择或环境约束，任务往往是交错进行的。因此，人类几乎不会在一个任务完全掌握之后才转移到下一个任务。为了探查类似人类的学习方法能给学习者带来多大程度的好处，我们提出了一种根据“学习进度”和能量消耗交错任务的方法。从机器学习的角度来看，我们的方法可以视为一种平衡学习性能与能量约束的多任务学习系统，同时模拟了生态上现实的人类任务学习。为了评估我们方法的有效性，我们在一个模拟的机器人学习设置中进行了实验，该设置中机器人在不同的情境中学习其动作的影响。实验结果表明，我们提出的方法在任务学习方面优于顺序学习方法，并且减少了学习任务的能量消耗。 

---
# Command A: An Enterprise-Ready Large Language Model 

**Title (ZH)**: 企业级大型语言模型 Command A 

**Authors**: Team Cohere, Aakanksha, Arash Ahmadian, Marwan Ahmed, Jay Alammar, Yazeed Alnumay, Sophia Althammer, Arkady Arkhangorodsky, Viraat Aryabumi, Dennis Aumiller, Raphaël Avalos, Zahara Aviv, Sammie Bae, Saurabh Baji, Alexandre Barbet, Max Bartolo, Björn Bebensee, Neeral Beladia, Walter Beller-Morales, Alexandre Bérard, Andrew Berneshawi, Anna Bialas, Phil Blunsom, Matt Bobkin, Adi Bongale, Sam Braun, Maxime Brunet, Samuel Cahyawijaya, David Cairuz, Jon Ander Campos, Cassie Cao, Kris Cao, Roman Castagné, Julián Cendrero, Leila Chan Currie, Yash Chandak, Diane Chang, Giannis Chatziveroglou, Hongyu Chen, Claire Cheng, Alexis Chevalier, Justin T. Chiu, Eugene Cho, Eugene Choi, Eujeong Choi, Tim Chung, Volkan Cirik, Ana Cismaru, Pierre Clavier, Henry Conklin, Lucas Crawhall-Stein, Devon Crouse, Andres Felipe Cruz-Salinas, Ben Cyrus, Daniel D'souza, Hugo Dalla-Torre, John Dang, William Darling, Omar Darwiche Domingues, Saurabh Dash, Antoine Debugne, Théo Dehaze, Shaan Desai, Joan Devassy, Rishit Dholakia, Kyle Duffy, Ali Edalati, Ace Eldeib, Abdullah Elkady, Sarah Elsharkawy, Irem Ergün, Beyza Ermis, Marzieh Fadaee, Boyu Fan, Lucas Fayoux, Yannis Flet-Berliac, Nick Frosst, Matthias Gallé, Wojciech Galuba, Utsav Garg, Matthieu Geist, Mohammad Gheshlaghi Azar, Seraphina Goldfarb-Tarrant, Tomas Goldsack, Aidan Gomez, Victor Machado Gonzaga, Nithya Govindarajan, Manoj Govindassamy, Nathan Grinsztajn, Nikolas Gritsch, Patrick Gu, Shangmin Guo, Kilian Haefeli, Rod Hajjar, Tim Hawes, Jingyi He, Sebastian Hofstätter, Sungjin Hong, Sara Hooker, Tom Hosking  

**Link**: [PDF](https://arxiv.org/pdf/2504.00698)  

**Abstract**: In this report we describe the development of Command A, a powerful large language model purpose-built to excel at real-world enterprise use cases. Command A is an agent-optimised and multilingual-capable model, with support for 23 languages of global business, and a novel hybrid architecture balancing efficiency with top of the range performance. It offers best-in-class Retrieval Augmented Generation (RAG) capabilities with grounding and tool use to automate sophisticated business processes. These abilities are achieved through a decentralised training approach, including self-refinement algorithms and model merging techniques. We also include results for Command R7B which shares capability and architectural similarities to Command A. Weights for both models have been released for research purposes. This technical report details our original training pipeline and presents an extensive evaluation of our models across a suite of enterprise-relevant tasks and public benchmarks, demonstrating excellent performance and efficiency. 

**Abstract (ZH)**: 本报告描述了Command A的开发，Command A是一款专为 excellence于真实企业应用场景设计的强大语言模型。Command A是一款优化过的代理模型，支持23种全球商业语言，并具备一种新颖的混合架构，平衡了效率与顶级性能。该模型具备最佳的检索增强生成（RAG）能力，支持接地和工具使用以自动化复杂业务流程。这些能力是通过去中心化的训练方法实现的，包括自我改进算法和模型合并技术。我们还介绍了与Command A具有能力和架构相似性的Command R7B的成果。这两个模型的权重均已发布用于研究目的。本技术报告详细介绍了我们的原始训练管道，并对模型在一系列企业相关任务和公共基准上的进行了广泛评估，展示了出色的表现和效率。 

---
# The HCI GenAI CO2ST Calculator: A Tool for Calculating the Carbon Footprint of Generative AI Use in Human-Computer Interaction Research 

**Title (ZH)**: 面向人机交互的生成式AI碳足迹计算器：一种计算生成式AI使用碳足迹的工具 

**Authors**: Nanna Inie, Jeanette Falk, Raghavendra Selvan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00692)  

**Abstract**: Increased usage of generative AI (GenAI) in Human-Computer Interaction (HCI) research induces a climate impact from carbon emissions due to energy consumption of the hardware used to develop and run GenAI models and systems. The exact energy usage and and subsequent carbon emissions are difficult to estimate in HCI research because HCI researchers most often use cloud-based services where the hardware and its energy consumption are hidden from plain view. The HCI GenAI CO2ST Calculator is a tool designed specifically for the HCI research pipeline, to help researchers estimate the energy consumption and carbon footprint of using generative AI in their research, either a priori (allowing for mitigation strategies or experimental redesign) or post hoc (allowing for transparent documentation of carbon footprint in written reports of the research). 

**Abstract (ZH)**: 生成式AI（GenAI）在人机交互（HCI）研究中的使用增加了因硬件能耗而导致的碳排放气候影响。由于HCI研究者通常使用云基服务，其中硬件及其能耗隐藏不见，因此在HCI研究中精确估计能耗和随之而来的碳排放具有困难。HCI生成式AI碳排放计算器是专门为HCI研究流程设计的工具，旨在帮助研究人员估算使用生成式AI在研究中的能耗和碳足迹，无论是事先（允许采取缓解策略或实验重设计）还是事后（允许在研究书面报告中透明地记录碳足迹）。 

---
# DynMoLE: Boosting Mixture of LoRA Experts Fine-Tuning with a Hybrid Routing Mechanism 

**Title (ZH)**: DynMoLE: 结合混合路由机制提升LoRA专家混合模型微调性能 

**Authors**: Dengchun Li, Naizheng Wang, Zihao Zhang, Haoyang Yin, Lei Duan, Meng Xiao, Mingjie Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00661)  

**Abstract**: Instruction-based fine-tuning of large language models (LLMs) has achieved remarkable success in various natural language processing (NLP) tasks. Parameter-efficient fine-tuning (PEFT) methods, such as Mixture of LoRA Experts (MoLE), combine the efficiency of Low-Rank Adaptation (LoRA) with the versatility of Mixture of Experts (MoE) models, demonstrating significant potential for handling multiple downstream tasks. However, the existing routing mechanisms for MoLE often involve a trade-off between computational efficiency and predictive accuracy, and they fail to fully address the diverse expert selection demands across different transformer layers. In this work, we propose DynMoLE, a hybrid routing strategy that dynamically adjusts expert selection based on the Tsallis entropy of the router's probability distribution. This approach mitigates router uncertainty, enhances stability, and promotes more equitable expert participation, leading to faster convergence and improved model performance. Additionally, we introduce an auxiliary loss based on Tsallis entropy to further guide the model toward convergence with reduced uncertainty, thereby improving training stability and performance. Our extensive experiments on commonsense reasoning benchmarks demonstrate that DynMoLE achieves substantial performance improvements, outperforming LoRA by 9.6% and surpassing the state-of-the-art MoLE method, MoLA, by 2.3%. We also conduct a comprehensive ablation study to evaluate the contributions of DynMoLE's key components. 

**Abstract (ZH)**: 基于指令的大型语言模型细调已在多种自然语言处理任务中取得了显著成功。参数效率细调（PEFT）方法，如Mixture of LoRA Experts (MoLE)，结合了LoRA的高效性和MoE模型的灵活性，展示了处理多种下游任务的巨大潜力。然而，MoLE现有的路由机制往往在计算效率和预测准确性之间存在权衡，并未能充分解决不同Transformer层的多样化专家选择需求。在本文中，我们提出DynMoLE，这是一种基于Tsallis熵的动态路由策略，根据路由器概率分布的Tsallis熵动态调整专家选择。这种方法减轻了路由器不确定性，提升了稳定性，并促进了更加公平的专家参与，从而加快了收敛速度并提高了模型性能。此外，我们引入了基于Tsallis熵的辅助损失，进一步引导模型以降低不确定性的方式向收敛目标发展，从而提高训练稳定性和性能。我们在常识推理基准上的 extensive 实验表明，DynMoLE取得了显著的性能提升，比LoRA高出9.6%，并优于最先进的MoLE方法MoLA 2.3%。我们还进行了全面的消融研究以评估DynMoLE关键组件的贡献。 

---
# Towards Adaptive AI Governance: Comparative Insights from the U.S., EU, and Asia 

**Title (ZH)**: 面向适应性AI治理：来自美国、欧盟和亚洲的比较洞察 

**Authors**: Vikram Kulothungan, Deepti Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2504.00652)  

**Abstract**: Artificial intelligence (AI) trends vary significantly across global regions, shaping the trajectory of innovation, regulation, and societal impact. This variation influences how different regions approach AI development, balancing technological progress with ethical and regulatory considerations. This study conducts a comparative analysis of AI trends in the United States (US), the European Union (EU), and Asia, focusing on three key dimensions: generative AI, ethical oversight, and industrial applications. The US prioritizes market-driven innovation with minimal regulatory constraints, the EU enforces a precautionary risk-based framework emphasizing ethical safeguards, and Asia employs state-guided AI strategies that balance rapid deployment with regulatory oversight. Although these approaches reflect different economic models and policy priorities, their divergence poses challenges to international collaboration, regulatory harmonization, and the development of global AI standards. To address these challenges, this paper synthesizes regional strengths to propose an adaptive AI governance framework that integrates risk-tiered oversight, innovation accelerators, and strategic alignment mechanisms. By bridging governance gaps, this study offers actionable insights for fostering responsible AI development while ensuring a balance between technological progress, ethical imperatives, and regulatory coherence. 

**Abstract (ZH)**: 全球不同地区的人工智能趋势差异显著，塑造了创新、监管和社会影响的轨迹。这种差异影响了不同地区在人工智能开发中的不同做法，平衡了科技进步与伦理和监管考量。本研究对比分析了美国、欧盟和亚洲的人工智能趋势，重点关注生成式人工智能、伦理监管和工业应用三个维度。美国以市场驱动的创新为主，法规约束较少；欧盟采用预防性的基于风险的框架，强调伦理保障；亚洲则采取由国家指导的人工智能战略，平衡快速部署与监管监督。尽管这些方法反映了不同的经济模式和政策优先事项，但它们之间的差异对国际协作、监管协调和全球人工智能标准的发展提出了挑战。为应对这些挑战，本文整合了区域优势，提出了一个适应性的人工智能治理框架，其中包括分级监管、创新加速器和战略对齐机制。通过弥合治理缺口，本研究提供了促进负责任的人工智能发展、确保在科技进步、伦理要求和监管一致性的平衡方面的实际建议。 

---
# Impact of Data Duplication on Deep Neural Network-Based Image Classifiers: Robust vs. Standard Models 

**Title (ZH)**: 基于深层神经网络的图像分类器中数据冗余的影响：稳健模型与标准模型的比较 

**Authors**: Alireza Aghabagherloo, Aydin Abadi, Sumanta Sarkar, Vishnu Asutosh Dasu, Bart Preneel  

**Link**: [PDF](https://arxiv.org/pdf/2504.00638)  

**Abstract**: The accuracy and robustness of machine learning models against adversarial attacks are significantly influenced by factors such as training data quality, model architecture, the training process, and the deployment environment. In recent years, duplicated data in training sets, especially in language models, has attracted considerable attention. It has been shown that deduplication enhances both training performance and model accuracy in language models. While the importance of data quality in training image classifier Deep Neural Networks (DNNs) is widely recognized, the impact of duplicated images in the training set on model generalization and performance has received little attention.
In this paper, we address this gap and provide a comprehensive study on the effect of duplicates in image classification. Our analysis indicates that the presence of duplicated images in the training set not only negatively affects the efficiency of model training but also may result in lower accuracy of the image classifier. This negative impact of duplication on accuracy is particularly evident when duplicated data is non-uniform across classes or when duplication, whether uniform or non-uniform, occurs in the training set of an adversarially trained model. Even when duplicated samples are selected in a uniform way, increasing the amount of duplication does not lead to a significant improvement in accuracy. 

**Abstract (ZH)**: 机器学习模型对抗 adversarial 攻击的准确性和鲁棒性受训练数据质量、模型架构、训练过程和部署环境等因素的影响。近年来，训练集中的重复数据，特别是在语言模型中，引起了广泛关注。研究表明，去重可以提高语言模型的训练性能和模型准确性。虽然图像分类深层神经网络（DNNs）训练数据质量的重要性得到广泛认可，但训练集中的重复图像对模型泛化能力和性能的影响却很少被关注。本文填补了这一空白，对图像分类中重复数据的影响进行了全面研究。我们的分析表明，训练集中的重复图像不仅负面影响了模型训练的效率，还可能导致图像分类器的准确性降低。这种重复数据对准确性的负面影响，在类别间重复数据非均匀分布或在对抗训练模型的训练集中出现均匀或非均匀重复数据时尤为明显。即使重复样本以均匀方式选择，增加重复数据的数量也不会显著提高准确性。 

---
# CNOT-Optimal Clifford Synthesis as SAT 

**Title (ZH)**: CNOT-最优克利福德合成作为SAT问题 

**Authors**: Irfansha Shaik, Jaco van de Pol  

**Link**: [PDF](https://arxiv.org/pdf/2504.00634)  

**Abstract**: Clifford circuit optimization is an important step in the quantum compilation pipeline. Major compilers employ heuristic approaches. While they are fast, their results are often suboptimal. Minimization of noisy gates, like 2-qubit CNOT gates, is crucial for practical computing. Exact approaches have been proposed to fill the gap left by heuristic approaches. Among these are SAT based approaches that optimize gate count or depth, but they suffer from scalability issues. Further, they do not guarantee optimality on more important metrics like CNOT count or CNOT depth. A recent work proposed an exhaustive search only on Clifford circuits in a certain normal form to guarantee CNOT count optimality. But an exhaustive approach cannot scale beyond 6 qubits.
In this paper, we incorporate search restricted to Clifford normal forms in a SAT encoding to guarantee CNOT count optimality. By allowing parallel plans, we propose a second SAT encoding that optimizes CNOT depth. By taking advantage of flexibility in SAT based approaches, we also handle connectivity restrictions in hardware platforms, and allow for qubit relabeling. We have implemented the above encodings and variations in our open source tool Q-Synth.
In experiments, our encodings significantly outperform existing SAT approaches on random Clifford circuits. We consider practical VQE and Feynman benchmarks to compare with TKET and Qiskit compilers. In all-to-all connectivity, we observe reductions up to 32.1% in CNOT count and 48.1% in CNOT depth. Overall, we observe better results than TKET in the CNOT count and depth. We also experiment with connectivity restrictions of major quantum platforms. Compared to Qiskit, we observe up to 30.3% CNOT count and 35.9% CNOT depth further reduction. 

**Abstract (ZH)**: Clifford 电路优化是量子编译管道中的一个关键步骤。主要编译器采用启发式方法。尽管这些方法速度快，但结果通常不理想。减少噪声门，如 2 腰 CNOT 门，对实际计算至关重要。已提出了精确方法以填补启发式方法的不足。其中一些方法基于 SAT 的优化，可以最小化门的数量或深度，但它们存在可扩展性问题。此外，它们在如 CNOT 数量或 CNOT 深度等更重要的指标上无法保证最优性。最近的一项工作提出了一种仅在特定正常形式的 Clifford 电路中进行穷举搜索的方法，以保证 CNOT 数量优化。然而，穷举方法无法扩展超过 6 个量子位。

本文中，我们在 SAT 编码中整合了仅针对 Clifford 正规形式的搜索，以保证 CNOT 数量优化。通过允许并行计划，我们提出了一种新的 SAT 编码方法，以优化 CNOT 深度。利用 SAT 方法的灵活性，我们还处理了硬件平台的连接性限制，并允许量子位重新标记。我们已在开源工具 Q-Synth 中实现了上述编码及其变体。

在实验中，我们的编码在随机 Clifford 电路的 SAT 方法中表现出显著的优异性能。我们考虑了实用的 VQE 和费曼基准与 TKET 和 Qiskit 编译器进行比较。在全连接情况下，我们观察到 CNOT 数量最多减少了 32.1%，CNOT 深度最多减少了 48.1%。总体而言，我们的 CNOT 数量和深度表现优于 TKET。我们还实验了主要量子平台的连接性限制。与 Qiskit 相比，我们观察到 CNOT 数量最多进一步减少了 30.3%，CNOT 深度最多进一步减少了 35.9%。 

---
# Feature Subset Weighting for Distance-based Supervised Learning through Choquet Integration 

**Title (ZH)**: 基于Choquet积分的特征子集加权距离导向监督学习 

**Authors**: Adnan Theerens, Yvan Saeys, Chris Cornelis  

**Link**: [PDF](https://arxiv.org/pdf/2504.00624)  

**Abstract**: This paper introduces feature subset weighting using monotone measures for distance-based supervised learning. The Choquet integral is used to define a distance metric that incorporates these weights. This integration enables the proposed distances to effectively capture non-linear relationships and account for interactions both between conditional and decision attributes and among conditional attributes themselves, resulting in a more flexible distance measure. In particular, we show how this approach ensures that the distances remain unaffected by the addition of duplicate and strongly correlated features. Another key point of this approach is that it makes feature subset weighting computationally feasible, since only $m$ feature subset weights should be calculated each time instead of calculating all feature subset weights ($2^m$), where $m$ is the number of attributes. Next, we also examine how the use of the Choquet integral for measuring similarity leads to a non-equivalent definition of distance. The relationship between distance and similarity is further explored through dual measures. Additionally, symmetric Choquet distances and similarities are proposed, preserving the classical symmetry between similarity and distance. Finally, we introduce a concrete feature subset weighting distance, evaluate its performance in a $k$-nearest neighbors (KNN) classification setting, and compare it against Mahalanobis distances and weighted distance methods. 

**Abstract (ZH)**: 本文介绍了一种使用单调测度进行特征子集加权的距离基监督学习方法。利用Choquet积分定义包含这些权重的距离度量，这种集成使得提出的距离能够有效地捕捉非线性关系并考虑条件属性之间以及决策属性和条件属性之间的相互作用，从而获得更灵活的距离度量。特别是，我们展示了这种方法确保在添加重复和强相关特征时距离不受影响。该方法的另一个关键点是它使特征子集加权在计算上可行，每次只需要计算$m$个特征子集权重，而不是计算所有特征子集权重($2^m$)，其中$m$是属性的数量。接下来，我们还探讨了使用Choquet积分衡量相似性导致的距离非等价定义。通过双测度进一步探讨了距离与相似性的关系。此外，提出了对称Choquet距离和相似性，保持了相似性和距离的经典对称性。最后，我们引入了一个具体的特征子集加权距离，在$k$-最近邻（KNN）分类设置中评估其性能，并将其与马氏距离和加权距离方法进行比较。 

---
# PLM4NDV: Minimizing Data Access for Number of Distinct Values Estimation with Pre-trained Language Models 

**Title (ZH)**: PLM4NDV：使用预训练语言模型最小化数据访问的数量唯一值估算 

**Authors**: Xianghong Xu, Xiao He, Tieying Zhang, Lei Zhang, Rui Shi, Jianjun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.00608)  

**Abstract**: Number of Distinct Values (NDV) estimation of a multiset/column is a basis for many data management tasks, especially within databases. Despite decades of research, most existing methods require either a significant amount of samples through uniform random sampling or access to the entire column to produce estimates, leading to substantial data access costs and potentially ineffective estimations in scenarios with limited data access. In this paper, we propose leveraging semantic information, i.e., schema, to address these challenges. The schema contains rich semantic information that can benefit the NDV estimation. To this end, we propose PLM4NDV, a learned method incorporating Pre-trained Language Models (PLMs) to extract semantic schema information for NDV estimation. Specifically, PLM4NDV leverages the semantics of the target column and the corresponding table to gain a comprehensive understanding of the column's meaning. By using the semantics, PLM4NDV reduces data access costs, provides accurate NDV estimation, and can even operate effectively without any data access. Extensive experiments on a large-scale real-world dataset demonstrate the superiority of PLM4NDV over baseline methods. Our code is available at this https URL. 

**Abstract (ZH)**: 基于语义信息的多集合/列的唯一值数量估计 

---
# Data Cleansing for GANs 

**Title (ZH)**: GANs的数据清洗 

**Authors**: Naoyuki Terashita, Hiroki Ohashi, Satoshi Hara  

**Link**: [PDF](https://arxiv.org/pdf/2504.00603)  

**Abstract**: As the application of generative adversarial networks (GANs) expands, it becomes increasingly critical to develop a unified approach that improves performance across various generative tasks. One effective strategy that applies to any machine learning task is identifying harmful instances, whose removal improves the performance. While previous studies have successfully estimated these harmful training instances in supervised settings, their approaches are not easily applicable to GANs. The challenge lies in two requirements of the previous approaches that do not apply to GANs. First, previous approaches require that the absence of a training instance directly affects the parameters. However, in the training for GANs, the instances do not directly affect the generator's parameters since they are only fed into the discriminator. Second, previous approaches assume that the change in loss directly quantifies the harmfulness of the instance to a model's performance, while common types of GAN losses do not always reflect the generative performance. To overcome the first challenge, we propose influence estimation methods that use the Jacobian of the generator's gradient with respect to the discriminator's parameters (and vice versa). Such a Jacobian represents the indirect effect between two models: how removing an instance from the discriminator's training changes the generator's parameters. Second, we propose an instance evaluation scheme that measures the harmfulness of each training instance based on how a GAN evaluation metric (e.g., Inception score) is expected to change by the instance's removal. Furthermore, we demonstrate that removing the identified harmful instances significantly improves the generative performance on various GAN evaluation metrics. 

**Abstract (ZH)**: 生成对抗网络（GANs）的应用扩展使得开发一种统一的方法来提高各种生成任务性能变得日益重要。一种适用于任何机器学习任务的有效策略是识别有害实例，通过移除这些实例可以提高性能。尽管先前的研究已经在监督设置中成功估计了这些有害的训练实例，但其方法不适用于GANs。先前方法的两个要求在GANs中并不适用。首先，先前方法要求训练实例的缺失直接影响模型参数，但在GANs的训练中，实例并不会直接影响生成器的参数，因为它们仅被输入到判别器中。其次，先前方法假设损失变化直接反映了实例对模型性能的有害性，而常见的GAN损失类型并不总是反映生成性能。为克服第一个挑战，我们提出了一种使用生成器梯度关于判别器参数的雅各宾矩阵（反之亦然）来估计影响的方法。此类雅各宾矩阵表示两个模型之间的间接影响：从判别器训练中移除一个实例如何改变生成器参数。其次，我们提出了一种实例评估方案，基于移除实例后预期的GAN评估指标（如Inception分数）的变化来衡量每个训练实例的有害性。此外，我们证明移除识别出的有害实例可以显著提高各种GAN评估指标的生成性能。 

---
# On the Consistency of Multilingual Context Utilization in Retrieval-Augmented Generation 

**Title (ZH)**: 多语言上下文利用在检索增强生成中的一致性研究 

**Authors**: Jirui Qi, Raquel Fernández, Arianna Bisazza  

**Link**: [PDF](https://arxiv.org/pdf/2504.00597)  

**Abstract**: Retrieval-augmented generation (RAG) with large language models (LLMs) has demonstrated strong performance in multilingual question-answering (QA) tasks by leveraging relevant passages retrieved from corpora. In multilingual RAG (mRAG), the retrieved passages can be written in languages other than that of the query entered by the user, making it challenging for LLMs to effectively utilize the provided information. Recent research suggests that retrieving passages from multilingual corpora can improve RAG performance, particularly for low-resource languages. However, the extent to which LLMs can leverage different kinds of multilingual contexts to generate accurate answers, *independently from retrieval quality*, remains understudied. In this paper, we conduct an extensive assessment of LLMs' ability to (i) make consistent use of a relevant passage regardless of its language, (ii) respond in the expected language, and (iii) focus on the relevant passage even when multiple `distracting' passages in different languages are provided in the context. Our experiments with four LLMs across three QA datasets covering a total of 48 languages reveal a surprising ability of LLMs to extract the relevant information from out-language passages, but a much weaker ability to formulate a full answer in the correct language. Our analysis, based on both accuracy and feature attribution techniques, further shows that distracting passages negatively impact answer quality regardless of their language. However, distractors in the query language exert a slightly stronger influence. Taken together, our findings deepen the understanding of how LLMs utilize context in mRAG systems, providing directions for future improvements. 

**Abstract (ZH)**: 基于大型语言模型的检索增强生成（RAG）在多语言问答任务中的表现通过利用从语料库中检索的相关段落得到了显著提升。在多语言RAG（mRAG）中，检索到的段落可以是与用户查询语言不同的语言，这使得大型语言模型难以有效利用提供的信息。最近的研究表明，从多语言语料库中检索段落可以提高RAG的表现，尤其是对于低资源语言。然而，大型语言模型利用不同类型的多语言上下文生成准确答案的能力，在不依赖于检索质量的情况下，仍需进一步研究。本文对四种大型语言模型在三个涵盖48种语言的问答数据集上的性能进行了广泛的评估，结果显示，大型语言模型具有从非本族语言段落中提取相关信息的令人惊讶的能力，但在生成正确语言的完整答案方面能力较弱。我们的分析表明，不相关段落无论其语言如何，都会负面影响答案质量，但查询语言的不相关段落影响更为显著。综上所述，我们的研究结果加深了对大型语言模型在多语言RAG系统中利用上下文的理解，并为未来改进指明了方向。 

---
# Enhancing Negation Awareness in Universal Text Embeddings: A Data-efficient and Computational-efficient Approach 

**Title (ZH)**: 提高通用文本嵌入中的否定意识：一种数据高效且计算高效的方法 

**Authors**: Hongliu Cao  

**Link**: [PDF](https://arxiv.org/pdf/2504.00584)  

**Abstract**: Negation plays an important role in various natural language processing tasks such as Natural Language Inference and Sentiment Analysis tasks. Numerous prior studies have found that contextual text embedding models such as BERT, ELMO, RoBERTa or XLNet face challenges in accurately understanding negation. Recent advancements in universal text embeddings have demonstrated superior performance over contextual text embeddings in various tasks. However, due to the bias in popular evaluation benchmarks, the negation awareness capacity of these models remains unclear. To bridge the gap in existing literature, an in-depth analysis is initiated in this work to study the negation awareness of cutting-edge universal text embedding models. Our findings reveal a significant lack of negation awareness in these models, often interpreting negated text pairs as semantically similar. To efficiently deal with the conflict that different tasks need different trade-offs between topic and negation information among other semantic information, a data-efficient and computational-efficient embedding re-weighting method is proposed without modifying the parameters of text embedding models. The proposed solution is able to improve text embedding models' negation awareness significantly on both simple negation understanding task and complex negation understanding task. Furthermore, the proposed solution can also significantly improve the negation awareness of Large Language Model based task-specific high dimensional universal text embeddings. 

**Abstract (ZH)**: 否定在自然语言推理和情感分析等自然语言处理任务中发挥着重要作用。尽管先前研究表明，诸如BERT、ELMO、RoBERTa或XLNet等上下文文本嵌入模型在理解否定方面面临挑战，近期的通用文本嵌入进展在各种任务中显示出更好的性能。然而，由于流行评估基准中的偏差，这些模型的否定感知能力仍不清楚。为弥合现有文献的差距，本研究深入分析了最新通用文本嵌入模型的否定感知能力。我们的发现揭示了这些模型在否定感知方面存在显著不足，经常将否定文本对错误地解释为语义相似。为进一步高效解决不同任务在主题和否定信息等其他语义信息之间需要不同权衡的问题，我们提出了一种无需修改文本嵌入模型参数的高效且计算效率高的嵌入重权方法。该解决方案能够显著提高文本嵌入模型在简单否定理解任务和复杂否定理解任务中的否定感知能力，并且还能显著提高基于大型语言模型的任务特定高维通用文本嵌入的否定感知能力。 

---
# High-Quality Pseudo-Label Generation Based on Visual Prompt Assisted Cloud Model Update 

**Title (ZH)**: 基于视觉提示辅助云模型更新的高质量伪标签生成 

**Authors**: Xinrun Xu, Qiuhong Zhang, Jianwen Yang, Zhanbiao Lian, Jin Yan, Zhiming Ding, Shan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00526)  

**Abstract**: Generating high-quality pseudo-labels on the cloud is crucial for cloud-edge object detection, especially in dynamic traffic monitoring where data distributions evolve. Existing methods often assume reliable cloud models, neglecting potential errors or struggling with complex distribution shifts. This paper proposes Cloud-Adaptive High-Quality Pseudo-label generation (CA-HQP), addressing these limitations by incorporating a learnable Visual Prompt Generator (VPG) and dual feature alignment into cloud model updates. The VPG enables parameter-efficient adaptation by injecting visual prompts, enhancing flexibility without extensive fine-tuning. CA-HQP mitigates domain discrepancies via two feature alignment techniques: global Domain Query Feature Alignment (DQFA) capturing scene-level shifts, and fine-grained Temporal Instance-Aware Feature Embedding Alignment (TIAFA) addressing instance variations. Experiments on the Bellevue traffic dataset demonstrate that CA-HQP significantly improves pseudo-label quality compared to existing methods, leading to notable performance gains for the edge model and showcasing CA-HQP's adaptation effectiveness. Ablation studies validate each component (DQFA, TIAFA, VPG) and the synergistic effect of combined alignment strategies, highlighting the importance of adaptive cloud updates and domain adaptation for robust object detection in evolving scenarios. CA-HQP provides a promising solution for enhancing cloud-edge object detection systems in real-world applications. 

**Abstract (ZH)**: 生成高质量的云端伪标签对于云端边缘对象检测至关重要，特别是在数据分布演变的动态交通监控中。现有的方法往往假设云模型可靠，忽视潜在的错误或难以处理复杂的分布偏移。本文提出了一种云自适应高质量伪标签生成方法（CA-HQP），通过引入可学习的视觉提示生成器（VPG）和双特征对齐，解决这些问题。VPG通过注入视觉提示实现参数高效的适应，增强灵活性而无需大量的微调。CA-HQP通过两种特征对齐技术来缓解领域差异：全局领域查询特征对齐（DQFA）捕捉场景级偏移，以及细粒度的时间感知实例特征嵌入对齐（TIAFA）解决实例变异。在Bellevue交通数据集上的实验表明，CA-HQP在伪标签质量上显著优于现有方法，显著提高了边缘模型的性能，并展示了CA-HQP的适应效果。消融研究验证了每个组件（DQFA、TIAFA、VPG）及其组合对齐策略的协同效应，突显了适应性云更新和领域适应对动态场景中稳健对象检测的重要性。CA-HQP为提升实际应用中的云端边缘对象检测系统提供了有前景的解决方案。 

---
# Automated detection of atomicity violations in large-scale systems 

**Title (ZH)**: 大规模系统中原子性违例的自动检测 

**Authors**: Hang He, Yixing Luo, Chengcheng Wan, Ting Su, Haiying Sun, Geguang Pu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00521)  

**Abstract**: Atomicity violations in interrupt-driven programs pose a significant threat to software safety in critical systems. These violations occur when the execution sequence of operations on shared resources is disrupted by asynchronous interrupts. Detecting atomicity violations is challenging due to the vast program state space, application-level code dependencies, and complex domain-specific knowledge. We propose Clover, a hybrid framework that integrates static analysis with large language model (LLM) agents to detect atomicity violations in real-world programs. Clover first performs static analysis to extract critical code snippets and operation information. It then initiates a multi-agent process, where the expert agent leverages domain-specific knowledge to detect atomicity violations, which are subsequently validated by the judge agent. Evaluations on RaceBench 2.1, SV-COMP, and RWIP demonstrate that Clover achieves a precision/recall of 92.3%/86.6%, outperforming existing approaches by 27.4-118.2% on F1-score. 

**Abstract (ZH)**: 中断驱动程序中的原子性违反对关键系统软件安全性构成了重大威胁。这些违反发生在共享资源操作的执行顺序被异步中断打断时。检测原子性违反由于巨大的程序状态空间、应用程序级代码依赖关系和复杂的领域特定知识而具有挑战性。我们提出了一种名为Clover的混合框架，该框架结合了静态分析和大型语言模型（LLM）代理以检测实际程序中的原子性违反。Clover首先进行静态分析以提取关键代码片段和操作信息。然后启动一个多代理过程，其中专家代理利用领域特定知识检测原子性违反，随后由裁判代理进行验证。在RaceBench 2.1、SV-COMP和RWIP上的评估表明，Clover的精确度/召回率达到了92.3%/86.6%，在F1分数上比现有方法高出27.4%-118.2%。 

---
# Training Frozen Feature Pyramid DINOv2 for Eyelid Measurements with Infinite Encoding and Orthogonal Regularization 

**Title (ZH)**: 训练冻结特征金字塔的DINOv2进行眼睑测量：无限编码与正交正则化 

**Authors**: Chun-Hung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.00515)  

**Abstract**: Accurate measurement of eyelid parameters such as Margin Reflex Distances (MRD1, MRD2) and Levator Function (LF) is critical in oculoplastic diagnostics but remains limited by manual, inconsistent methods. This study evaluates deep learning models: SE-ResNet, EfficientNet, and the vision transformer-based DINOv2 for automating these measurements using smartphone-acquired images. We assess performance across frozen and fine-tuned settings, using MSE, MAE, and R2 metrics. DINOv2, pretrained through self-supervised learning, demonstrates superior scalability and robustness, especially under frozen conditions ideal for mobile deployment. Lightweight regressors such as MLP and Deep Ensemble offer high precision with minimal computational overhead. To address class imbalance and improve generalization, we integrate focal loss, orthogonal regularization, and binary encoding strategies. Our results show that DINOv2 combined with these enhancements delivers consistent, accurate predictions across all tasks, making it a strong candidate for real-world, mobile-friendly clinical applications. This work highlights the potential of foundation models in advancing AI-powered ophthalmic care. 

**Abstract (ZH)**: 准确测量眼睑参数（如MRD1、MRD2和LEV功能）对于眼睑整形诊断至关重要，但目前仍受限于手动且不一致的方法。本研究评估了SE-ResNet、EfficientNet和基于视觉变换器的DINOv2等深度学习模型，以利用手机拍摄的图像自动进行这些测量。我们使用均方误差、平均绝对误差和相关系数对不同固定和微调设置下的模型性能进行了评估。通过自监督学习预训练的DINOv2在固定条件下表现出色，具备更好的可扩展性和鲁棒性，特别适合移动部署。轻量级回归模型如MLP和深度集成模型提供了高精度且计算成本低的优势。为解决类别不平衡并提高泛化能力，我们引入了焦点损失、正交正则化和二进制编码策略。实验结果表明，结合这些增强后的DINOv2模型在所有任务中都能提供一致且准确的预测，使其成为面向实际应用场景的移动友好型临床应用的有力候选。本研究突显了基础模型在推进AI辅助眼科护理方面的潜力。 

---
# Operator Learning with Domain Decomposition for Geometry Generalization in PDE Solving 

**Title (ZH)**: 基于领域分解的运算器学习在偏微分方程求解中的几何泛化 

**Authors**: Jianing Huang, Kaixuan Zhang, Youjia Wu, Ze Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.00510)  

**Abstract**: Neural operators have become increasingly popular in solving \textit{partial differential equations} (PDEs) due to their superior capability to capture intricate mappings between function spaces over complex domains. However, the data-hungry nature of operator learning inevitably poses a bottleneck for their widespread applications. At the core of the challenge lies the absence of transferability of neural operators to new geometries. To tackle this issue, we propose operator learning with domain decomposition, a local-to-global framework to solve PDEs on arbitrary geometries. Under this framework, we devise an iterative scheme \textit{Schwarz Neural Inference} (SNI). This scheme allows for partitioning of the problem domain into smaller subdomains, on which local problems can be solved with neural operators, and stitching local solutions to construct a global solution. Additionally, we provide a theoretical analysis of the convergence rate and error bound. We conduct extensive experiments on several representative PDEs with diverse boundary conditions and achieve remarkable geometry generalization compared to alternative methods. These analysis and experiments demonstrate the proposed framework's potential in addressing challenges related to geometry generalization and data efficiency. 

**Abstract (ZH)**: 神经算子在解决偏微分方程（PDEs）方面的应用越来越受欢迎，得益于其在复杂域上函数空间之间捕获复杂映射的优异能力。然而，算子学习对数据的高需求不可避免地为其广泛应用设置了瓶颈。挑战的核心在于神经算子在新几何结构上的不可转移性。为了解决这一问题，我们提出了基于域分解的算子学习方法，这是一种将问题域分解为小子域、并在子域上使用神经算子求解局部问题，然后再将局部解拼接成全局解的局部到全局框架。在此框架下，我们设计了一种迭代方案—— Schwarz 神经推理（SNI）。此外，我们还提供了收敛速率和误差界的相关理论分析。我们对多个具有不同边界条件的代表性 PDE 进行了广泛的实验，并在几何泛化方面取得了显著效果，优于其他替代方法。这些分析和实验表明，所提出框架在应对几何泛化和数据效率相关挑战方面具有潜在应用价值。 

---
# Enhancing stroke disease classification through machine learning models via a novel voting system by feature selection techniques 

**Title (ZH)**: 通过特征选择技术实现的新型投票系统增强中风疾病分类的机器学习模型 

**Authors**: Mahade Hasan, Farhana Yasmin, Md. Mehedi Hassan, Xue Yu, Soniya Yeasmin, Herat Joshi, Sheikh Mohammed Shariful Islam  

**Link**: [PDF](https://arxiv.org/pdf/2504.00485)  

**Abstract**: Heart disease remains a leading cause of mortality and morbidity worldwide, necessitating the development of accurate and reliable predictive models to facilitate early detection and intervention. While state of the art work has focused on various machine learning approaches for predicting heart disease, but they could not able to achieve remarkable accuracy. In response to this need, we applied nine machine learning algorithms XGBoost, logistic regression, decision tree, random forest, k-nearest neighbors (KNN), support vector machine (SVM), gaussian naïve bayes (NB gaussian), adaptive boosting, and linear regression to predict heart disease based on a range of physiological indicators. Our approach involved feature selection techniques to identify the most relevant predictors, aimed at refining the models to enhance both performance and interpretability. The models were trained, incorporating processes such as grid search hyperparameter tuning, and cross-validation to minimize overfitting. Additionally, we have developed a novel voting system with feature selection techniques to advance heart disease classification. Furthermore, we have evaluated the models using key performance metrics including accuracy, precision, recall, F1-score, and the area under the receiver operating characteristic curve (ROC AUC). Among the models, XGBoost demonstrated exceptional performance, achieving 99% accuracy, precision, F1-Score, 98% recall, and 100% ROC AUC. This study offers a promising approach to early heart disease diagnosis and preventive healthcare. 

**Abstract (ZH)**: 心臟疾病仍然是全球 Leading 的致死和致病主要因素，亟需开发准确可靠的预测模型以促进早期检测和干预。尽管最先进的研究集中在各种机器学习方法来预测心臟疾病，但它们未能达到显著的准确性。为应对这一需求，我们应用了九种机器学习算法（XGBoost、逻辑回归、决策树、随机森林、K-近邻（KNN）、支持向量机（SVM）、高斯朴素贝叶斯（Gaussian Naïve Bayes）、自适应提升和线性回归），基于一系列生理指标预测心臟疾病。我们的方法包括特征选择技术，以识别最相关的预测因子，旨在优化模型以提高性能和可解释性。模型经过训练，并采用网格搜索超参数调优和交叉验证等过程，以减少过拟合。此外，我们还开发了一种新颖的投票系统并结合特征选择技术，以推进心臟疾病分类。进一步地，我们使用关键性能指标（包括准确率、精确率、召回率、F1分数和受试者操作特征曲线下的面积（ROC AUC））来评估模型。其中，XGBoost表现出色，实现了99%的准确率、精确率、F1分数，98%的召回率和100%的ROC AUC。本研究提供了早期心臟疾病诊断和预防保健的有前途的方法。 

---
# Memorizing is Not Enough: Deep Knowledge Injection Through Reasoning 

**Title (ZH)**: 记忆不足：通过推理注入深度知识 

**Authors**: Ruoxi Xu, Yunjie Ji, Boxi Cao, Yaojie Lu, Hongyu Lin, Xianpei Han, Ben He, Yingfei Sun, Xiangang Li, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.00472)  

**Abstract**: Although large language models (LLMs) excel in knowledge recall and reasoning, their static nature leads to outdated information as the real world evolves or when adapting to domain-specific knowledge, highlighting the need for effective knowledge injection. However, current research on knowledge injection remains superficial, mainly focusing on knowledge memorization and retrieval. This paper proposes a four-tier knowledge injection framework that systematically defines the levels of knowledge injection: memorization, retrieval, reasoning, and association. Based on this framework, we introduce DeepKnowledge, a synthetic experimental testbed designed for fine-grained evaluation of the depth of knowledge injection across three knowledge types (novel, incremental, and updated). We then explore various knowledge injection scenarios and evaluate the depth of knowledge injection for each scenario on the benchmark. Experimental results reveal key factors to reach each level of knowledge injection for LLMs and establish a mapping between the levels of knowledge injection and the corresponding suitable injection methods, aiming to provide a comprehensive approach for efficient knowledge injection across various levels. 

**Abstract (ZH)**: 尽管大型语言模型在知识回忆和推理方面表现出色，但由于其静态特性，在现实世界发展或适应领域特定知识时会出现过时信息的问题，凸显了有效知识注入的必要性。然而，当前的知识注入研究仍然停留在表面，主要集中在知识的记忆和检索。本文提出了一种四层知识注入框架，系统定义了知识注入的四个层级：记忆、检索、推理和关联。基于此框架，我们介绍了DeepKnowledge，这是一种合成实验测试床，用于对不同类型知识（新颖、增量和更新）的知识注入深度进行细粒度评估。随后，我们探索了各种知识注入场景，并在基准测试上评估了每个场景的知识注入深度。实验结果揭示了达到大型语言模型每个层级知识注入的关键因素，并建立了知识注入层级与相应适当注入方法之间的映射，旨在为不同层级的有效知识注入提供全面的方法。 

---
# Learning-Based Approximate Nonlinear Model Predictive Control Motion Cueing 

**Title (ZH)**: 基于学习的近似非线性模型预测控制运动模拟 

**Authors**: Camilo Gonzalez Arango, Houshyar Asadi, Mohammad Reza Chalak Qazani, Chee Peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2504.00469)  

**Abstract**: Motion Cueing Algorithms (MCAs) encode the movement of simulated vehicles into movement that can be reproduced with a motion simulator to provide a realistic driving experience within the capabilities of the machine. This paper introduces a novel learning-based MCA for serial robot-based motion simulators. Building on the differentiable predictive control framework, the proposed method merges the advantages of Nonlinear Model Predictive Control (NMPC) - notably nonlinear constraint handling and accurate kinematic modeling - with the computational efficiency of machine learning. By shifting the computational burden to offline training, the new algorithm enables real-time operation at high control rates, thus overcoming the key challenge associated with NMPC-based motion cueing. The proposed MCA incorporates a nonlinear joint-space plant model and a policy network trained to mimic NMPC behavior while accounting for joint acceleration, velocity, and position limits. Simulation experiments across multiple motion cueing scenarios showed that the proposed algorithm performed on par with a state-of-the-art NMPC-based alternative in terms of motion cueing quality as quantified by the RMSE and correlation coefficient with respect to reference signals. However, the proposed algorithm was on average 400 times faster than the NMPC baseline. In addition, the algorithm successfully generalized to unseen operating conditions, including motion cueing scenarios on a different vehicle and real-time physics-based simulations. 

**Abstract (ZH)**: 基于学习的串联机器人运动模拟器运动引导算法 

---
# MetaLoRA: Tensor-Enhanced Adaptive Low-Rank Fine-tuning 

**Title (ZH)**: MetaLoRA: 张量增强自适应低秩微调 

**Authors**: Maolin Wang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.00460)  

**Abstract**: There has been a significant increase in the deployment of neural network models, presenting substantial challenges in model adaptation and fine-tuning. Efficient adaptation is crucial in maintaining model performance across diverse tasks and domains. While Low-Rank Adaptation (LoRA) has emerged as a promising parameter-efficient fine-tuning method, its fixed parameter nature limits its ability to handle dynamic task requirements effectively. Adapting models to new tasks can be challenging due to the need for extensive fine-tuning. Current LoRA variants primarily focus on general parameter reduction while overlooking the importance of dynamic parameter adjustment and meta-learning capabilities. Moreover, existing approaches mainly address static adaptations, neglecting the potential benefits of task-aware parameter generation in handling diverse task distributions. To address these limitations, this Ph.D. research proposes a LoRA generation approach to model task relationships and introduces MetaLoRA, a novel parameter-efficient adaptation framework incorporating meta-learning principles. This work develops a comprehensive architecture that integrates meta-parameter generation with adaptive low-rank decomposition, enabling efficient handling of both task-specific and task-agnostic features. MetaLoRA accurately captures task patterns by incorporating meta-learning mechanisms and dynamic parameter adjustment strategies. To our knowledge, this research represents the first attempt to provide a meta-learning enhanced LoRA variant, offering improved adaptation capability while maintaining computational efficiency in model fine-tuning. 

**Abstract (ZH)**: 低秩适应增强的元学习框架：MetaLoRA 

---
# Distilling Multi-view Diffusion Models into 3D Generators 

**Title (ZH)**: 蒸馏多视图扩散模型为3D生成器 

**Authors**: Hao Qin, Luyuan Chen, Ming Kong, Mengxu Lu, Qiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00457)  

**Abstract**: We introduce DD3G, a formulation that Distills a multi-view Diffusion model (MV-DM) into a 3D Generator using gaussian splatting. DD3G compresses and integrates extensive visual and spatial geometric knowledge from the MV-DM by simulating its ordinary differential equation (ODE) trajectory, ensuring the distilled generator generalizes better than those trained solely on 3D data. Unlike previous amortized optimization approaches, we align the MV-DM and 3D generator representation spaces to transfer the teacher's probabilistic flow to the student, thus avoiding inconsistencies in optimization objectives caused by probabilistic sampling. The introduction of probabilistic flow and the coupling of various attributes in 3D Gaussians introduce challenges in the generation process. To tackle this, we propose PEPD, a generator consisting of Pattern Extraction and Progressive Decoding phases, which enables efficient fusion of probabilistic flow and converts a single image into 3D Gaussians within 0.06 seconds. Furthermore, to reduce knowledge loss and overcome sparse-view supervision, we design a joint optimization objective that ensures the quality of generated samples through explicit supervision and implicit verification. Leveraging existing 2D generation models, we compile 120k high-quality RGBA images for distillation. Experiments on synthetic and public datasets demonstrate the effectiveness of our method. Our project is available at: this https URL 

**Abstract (ZH)**: DD3G：一种通过高斯绘制将多视图扩散模型提炼为3D生成器的公式 

---
# No Free Lunch with Guardrails 

**Title (ZH)**: 有护栏也不能免于困境 

**Authors**: Divyanshu Kumar, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00441)  

**Abstract**: As large language models (LLMs) and generative AI become widely adopted, guardrails have emerged as a key tool to ensure their safe use. However, adding guardrails isn't without tradeoffs; stronger security measures can reduce usability, while more flexible systems may leave gaps for adversarial attacks. In this work, we explore whether current guardrails effectively prevent misuse while maintaining practical utility. We introduce a framework to evaluate these tradeoffs, measuring how different guardrails balance risk, security, and usability, and build an efficient guardrail.
Our findings confirm that there is no free lunch with guardrails; strengthening security often comes at the cost of usability. To address this, we propose a blueprint for designing better guardrails that minimize risk while maintaining usability. We evaluate various industry guardrails, including Azure Content Safety, Bedrock Guardrails, OpenAI's Moderation API, Guardrails AI, Nemo Guardrails, and our own custom-built guardrails. Additionally, we assess how LLMs like GPT-4o, Gemini 2.0-Flash, Claude 3.5-Sonnet, and Mistral Large-Latest respond under different system prompts, including simple prompts, detailed prompts, and detailed prompts with chain-of-thought (CoT) reasoning. Our study provides a clear comparison of how different guardrails perform, highlighting the challenges in balancing security and usability. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）和生成式AI的广泛应用，护栏已成为确保其安全使用的关键工具。然而，增加护栏并非没有取舍；增强的安全措施可能会降低易用性，而更灵活的系统可能会留下对抗性攻击的漏洞。在本研究中，我们探讨当前护栏是否能够有效防止滥用的同时保持其实用性。我们引入了一种评估这些取舍的框架，衡量不同护栏如何平衡风险、安全和易用性，并构建了一个高效的护栏。 

---
# Suite-IN++: A FlexiWear BodyNet Integrating Global and Local Motion Features from Apple Suite for Robust Inertial Navigation 

**Title (ZH)**: Suite-IN++：结合Apple Suite全局和局部 motion特征的柔性穿戴BodyNet稳健惯性导航 

**Authors**: Lan Sun, Songpengcheng Xia, Jiarui Yang, Ling Pei  

**Link**: [PDF](https://arxiv.org/pdf/2504.00438)  

**Abstract**: The proliferation of wearable technology has established multi-device ecosystems comprising smartphones, smartwatches, and headphones as critical enablers for ubiquitous pedestrian localization. However, traditional pedestrian dead reckoning (PDR) struggles with diverse motion modes, while data-driven methods, despite improving accuracy, often lack robustness due to their reliance on a single-device setup. Therefore, a promising solution is to fully leverage existing wearable devices to form a flexiwear bodynet for robust and accurate pedestrian localization. This paper presents Suite-IN++, a deep learning framework for flexiwear bodynet-based pedestrian localization. Suite-IN++ integrates motion data from wearable devices on different body parts, using contrastive learning to separate global and local motion features. It fuses global features based on the data reliability of each device to capture overall motion trends and employs an attention mechanism to uncover cross-device correlations in local features, extracting motion details helpful for accurate localization. To evaluate our method, we construct a real-life flexiwear bodynet dataset, incorporating Apple Suite (iPhone, Apple Watch, and AirPods) across diverse walking modes and device configurations. Experimental results demonstrate that Suite-IN++ achieves superior localization accuracy and robustness, significantly outperforming state-of-the-art models in real-life pedestrian tracking scenarios. 

**Abstract (ZH)**: 穿戴设备技术的普及已经建立了以智能手机、智能手表和耳机为核心的多设备生态系统，这些设备是实现泛在行人定位的关键使能器。然而，传统的行人航位推算（PDR）难以应对多种运动模式，而基于数据驱动的方法尽管提高了精度，但由于依赖单一设备设置，往往会缺乏鲁棒性。因此，一个有前景的解决方案是充分利用现有的穿戴设备，构建一个灵活穿戴体络（flexiwear bodynet），以实现稳健且精确的行人定位。本文提出了Suite-IN++，这是一种基于灵活穿戴体络的行人定位深度学习框架。Suite-IN++整合了不同身体部位穿戴设备的运动数据，利用对比学习分离全局和局部运动特征。它基于每个设备数据的可靠性融合全局特征，以捕捉整体运动趋势，并采用注意力机制揭示局部特征中的跨设备关联，提取有助于精确定位的运动细节。为了评估我们的方法，我们构建了一个实际生活中的灵活穿戴体络数据集，该数据集包含了在多种行走模式和设备配置下的Apple Suite（iPhone、Apple Watch和AirPods）。实验结果表明，Suite-IN++在实现卓越的定位精度和鲁棒性方面显著优于现有最先进的模型，在实际生活中的行人跟踪场景中表现优异。 

---
# LLM-Assisted Proactive Threat Intelligence for Automated Reasoning 

**Title (ZH)**: LLM辅助主动威胁情报以实现自动化推理 

**Authors**: Shuva Paul, Farhad Alemi, Richard Macwan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00428)  

**Abstract**: Successful defense against dynamically evolving cyber threats requires advanced and sophisticated techniques. This research presents a novel approach to enhance real-time cybersecurity threat detection and response by integrating large language models (LLMs) and Retrieval-Augmented Generation (RAG) systems with continuous threat intelligence feeds. Leveraging recent advancements in LLMs, specifically GPT-4o, and the innovative application of RAG techniques, our approach addresses the limitations of traditional static threat analysis by incorporating dynamic, real-time data sources. We leveraged RAG to get the latest information in real-time for threat intelligence, which is not possible in the existing GPT-4o model. We employ the Patrowl framework to automate the retrieval of diverse cybersecurity threat intelligence feeds, including Common Vulnerabilities and Exposures (CVE), Common Weakness Enumeration (CWE), Exploit Prediction Scoring System (EPSS), and Known Exploited Vulnerabilities (KEV) databases, and integrate these with the all-mpnet-base-v2 model for high-dimensional vector embeddings, stored and queried in Milvus. We demonstrate our system's efficacy through a series of case studies, revealing significant improvements in addressing recently disclosed vulnerabilities, KEVs, and high-EPSS-score CVEs compared to the baseline GPT-4o. This work not only advances the role of LLMs in cybersecurity but also establishes a robust foundation for the development of automated intelligent cyberthreat information management systems, addressing crucial gaps in current cybersecurity practices. 

**Abstract (ZH)**: 成功防御动态演变的网络威胁需要先进的复杂技术。本研究提出了一种通过将大型语言模型（LLMs）和检索增强生成（RAG）系统与持续的威胁情报流相结合来增强实时网络安全威胁检测和响应的新方法。利用最新的LLM发展，特别是GPT-4o，并创新地应用RAG技术，该方法通过集成动态和实时数据源解决了传统静态威胁分析的局限性。我们利用RAG实现实时获取最新的威胁情报，这是现有的GPT-4o模型所不具备的能力。我们使用Patrowl框架自动化了多种网络安全威胁情报流的检索，包括通用漏洞和暴露（CVE）、通用弱点枚举（CWE）、漏洞利用预测评分系统（EPSS）和已知利用漏洞（KEV）数据库，并将这些数据与all-mpnet-base-v2模型结合，用于高维向量嵌入，并在Milvus中存储和查询。我们通过一系列案例研究展示了该系统的有效性，相对于基线GPT-4o，显著提高了对最近披露的漏洞、KEVs和高EPSS评分CVE的处理能力。本研究不仅推动了LLMs在网络安全领域的应用，还为自动智能网络威胁信息管理系统的开发奠定了坚实的基础，填补了当前网络安全实践中的关键空白。 

---
# Multimodal LLMs for OCR, OCR Post-Correction, and Named Entity Recognition in Historical Documents 

**Title (ZH)**: 多模态LLM在历史文档中的OCR、OCR后修正及命名实体识别 

**Authors**: Gavin Greif, Niclas Griesshaber, Robin Greif  

**Link**: [PDF](https://arxiv.org/pdf/2504.00414)  

**Abstract**: We explore how multimodal Large Language Models (mLLMs) can help researchers transcribe historical documents, extract relevant historical information, and construct datasets from historical sources. Specifically, we investigate the capabilities of mLLMs in performing (1) Optical Character Recognition (OCR), (2) OCR Post-Correction, and (3) Named Entity Recognition (NER) tasks on a set of city directories published in German between 1754 and 1870. First, we benchmark the off-the-shelf transcription accuracy of both mLLMs and conventional OCR models. We find that the best-performing mLLM model significantly outperforms conventional state-of-the-art OCR models and other frontier mLLMs. Second, we are the first to introduce multimodal post-correction of OCR output using mLLMs. We find that this novel approach leads to a drastic improvement in transcription accuracy and consistently produces highly accurate transcriptions (<1% CER), without any image pre-processing or model fine-tuning. Third, we demonstrate that mLLMs can efficiently recognize entities in transcriptions of historical documents and parse them into structured dataset formats. Our findings provide early evidence for the long-term potential of mLLMs to introduce a paradigm shift in the approaches to historical data collection and document transcription. 

**Abstract (ZH)**: 多模态大型语言模型在历史文档转录、信息提取和数据集构建中的应用：以1754年至1870年间出版的德语城市目录为例 

---
# Semantic Mastery: Enhancing LLMs with Advanced Natural Language Understanding 

**Title (ZH)**: 语义掌握：通过高级自然语言理解增强LLMs 

**Authors**: Mohanakrishnan Hariharan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00409)  

**Abstract**: Large language models (LLMs) have greatly improved their capability in performing NLP tasks. However, deeper semantic understanding, contextual coherence, and more subtle reasoning are still difficult to obtain. The paper discusses state-of-the-art methodologies that advance LLMs with more advanced NLU techniques, such as semantic parsing, knowledge integration, and contextual reinforcement learning. We analyze the use of structured knowledge graphs, retrieval-augmented generation (RAG), and fine-tuning strategies that match models with human-level understanding. Furthermore, we address the incorporation of transformer-based architectures, contrastive learning, and hybrid symbolic-neural methods that address problems like hallucinations, ambiguity, and inconsistency in the factual perspectives involved in performing complex NLP tasks, such as question-answering text summarization and dialogue generation. Our findings show the importance of semantic precision for enhancing AI-driven language systems and suggest future research directions to bridge the gap between statistical language models and true natural language understanding. 

**Abstract (ZH)**: 大型语言模型（LLMs）在执行NLP任务方面的能力有显著提升，但仍难以获得更深层次的语义理解、上下文连贯性和更为细致的推理能力。本文讨论了采用更先进的自然语言理解技术（如语义解析、知识集成和上下文强化学习）来推进LLMs的前沿方法。我们分析了结构化知识图谱的应用、检索增强生成（RAG）以及与人类级理解相匹配的微调策略。此外，我们还探讨了基于变压器的架构、对比学习以及混合符号-神经方法在解决复杂NLP任务中的幻觉、模糊性和事实一致性问题方面的作用。我们的研究结果强调了语义精确性对于增强AI驱动的语言系统的关键作用，并提出了未来研究方向，以弥合统计语言模型与真正自然语言理解之间的差距。 

---
# From Intuition to Understanding: Using AI Peers to Overcome Physics Misconceptions 

**Title (ZH)**: 从直觉到理解：使用AI同伴克服物理misconceptions 

**Authors**: Ruben Weijers, Denton Wu, Hannah Betts, Tamara Jacod, Yuxiang Guan, Vidya Sujaya, Kushal Dev, Toshali Goel, William Delooze, Reihaneh Rabbany, Ying Wu, Jean-François Godbout, Kellin Pelrine  

**Link**: [PDF](https://arxiv.org/pdf/2504.00408)  

**Abstract**: Generative AI has the potential to transform personalization and accessibility of education. However, it raises serious concerns about accuracy and helping students become independent critical thinkers. In this study, we designed a helpful AI "Peer" to help students correct fundamental physics misconceptions related to Newtonian mechanic concepts. In contrast to approaches that seek near-perfect accuracy to create an authoritative AI tutor or teacher, we directly inform students that this AI can answer up to 40% of questions incorrectly. In a randomized controlled trial with 165 students, those who engaged in targeted dialogue with the AI Peer achieved post-test scores that were, on average, 10.5 percentage points higher - with over 20 percentage points higher normalized gain - than a control group that discussed physics history. Qualitative feedback indicated that 91% of the treatment group's AI interactions were rated as helpful. Furthermore, by comparing student performance on pre- and post-test questions about the same concept, along with experts' annotations of the AI interactions, we find initial evidence suggesting the improvement in performance does not depend on the correctness of the AI. With further research, the AI Peer paradigm described here could open new possibilities for how we learn, adapt to, and grow with AI. 

**Abstract (ZH)**: 生成式AI有潜力变革教育的个性化和可及性，但同时也引发了关于准确性和帮助学生培养独立批判性思维的严重关切。在本研究中，我们设计了一个有益的人工智能“同伴”，帮助学生纠正与牛顿力学概念相关的物理误解。与寻求近乎完美准确度以创建权威的人工智能导师或教师的方法不同，我们直接告知学生，该人工智能可能会错误回答多达40%的问题。在涉及165名学生的随机对照试验中，与对照组讨论物理历史相比，与AI同伴进行目标对话的学生在测试中的平均得分高出10.5个百分点，标准化增益高出20个百分点以上。定性反馈显示，治疗组中有91%的人工智能互动被认为是有帮助的。此外，通过比较学生在相同概念上的前后测表现以及专家对人工智能互动的注释，我们发现了初步证据，表明性能的提升并不依赖于人工智能的正确性。通过进一步研究，此处描述的AI同伴范式有可能为如何学习、适应和成长于人工智能开辟新的可能性。 

---
# VerifiAgent: a Unified Verification Agent in Language Model Reasoning 

**Title (ZH)**: VerifiAgent：语言模型推理中的统一验证代理 

**Authors**: Jiuzhou Han, Wray Buntine, Ehsan Shareghi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00406)  

**Abstract**: Large language models demonstrate remarkable reasoning capabilities but often produce unreliable or incorrect responses. Existing verification methods are typically model-specific or domain-restricted, requiring significant computational resources and lacking scalability across diverse reasoning tasks. To address these limitations, we propose VerifiAgent, a unified verification agent that integrates two levels of verification: meta-verification, which assesses completeness and consistency in model responses, and tool-based adaptive verification, where VerifiAgent autonomously selects appropriate verification tools based on the reasoning type, including mathematical, logical, or commonsense reasoning. This adaptive approach ensures both efficiency and robustness across different verification scenarios. Experimental results show that VerifiAgent outperforms baseline verification methods (e.g., deductive verifier, backward verifier) among all reasoning tasks. Additionally, it can further enhance reasoning accuracy by leveraging feedback from verification results. VerifiAgent can also be effectively applied to inference scaling, achieving better results with fewer generated samples and costs compared to existing process reward models in the mathematical reasoning domain. Code is available at this https URL 

**Abstract (ZH)**: 大规模语言模型展示了卓越的推理能力，但往往会产生不可靠或错误的响应。现有的验证方法通常是模型特定的或领域受限的，需要大量的计算资源，并且缺乏跨多样化推理任务的扩展性。为解决这些局限性，我们提出VerifiAgent，这是一种统一的验证代理，整合了两个层次的验证：元验证，评估模型响应的完整性和一致性；以及基于工具的自适应验证，VerifiAgent根据推理类型（包括数学、逻辑或常识推理）自主选择合适的验证工具。这种自适应方法确保了在不同的验证场景中兼具效率和鲁棒性。实验结果显示，VerifiAgent在所有推理任务中优于基线验证方法（如演绎验证器、回溯验证器）。此外，它还可以通过利用验证结果的反馈进一步提升推理准确性。在数学推理领域，VerifiAgent还可以有效应用于推理扩展，相较于现有的过程奖励模型，能在更少的生成样本和成本下取得更好的结果。代码可在此处访问。 

---
# Beyond Wide-Angle Images: Unsupervised Video Portrait Correction via Spatiotemporal Diffusion Adaptation 

**Title (ZH)**: 超越宽视角图像：基于空时扩散适应的无监督视频人像矫正 

**Authors**: Wenbo Nie, Lang Nie, Chunyu Lin, Jingwen Chen, Ke Xing, Jiyuan Wang, Yao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.00401)  

**Abstract**: Wide-angle cameras, despite their popularity for content creation, suffer from distortion-induced facial stretching-especially at the edge of the lens-which degrades visual appeal. To address this issue, we propose an image portrait correction framework using diffusion models named ImagePD. It integrates the long-range awareness of transformer and multi-step denoising of diffusion models into a unified framework, achieving global structural robustness and local detail refinement. Besides, considering the high cost of obtaining video labels, we then repurpose ImagePD for unlabeled wide-angle videos (termed VideoPD), by spatiotemporal diffusion adaption with spatial consistency and temporal smoothness constraints. For the former, we encourage the denoised image to approximate pseudo labels following the wide-angle distortion distribution pattern, while for the latter, we derive rectification trajectories with backward optical flows and smooth them. Compared with ImagePD, VideoPD maintains high-quality facial corrections in space and mitigates the potential temporal shakes sequentially. Finally, to establish an evaluation benchmark and train the framework, we establish a video portrait dataset with a large diversity in people number, lighting conditions, and background. Experiments demonstrate that the proposed methods outperform existing solutions quantitatively and qualitatively, contributing to high-fidelity wide-angle videos with stable and natural portraits. The codes and dataset will be available. 

**Abstract (ZH)**: 宽视角相机尽管在内容创作中很流行，但由于镜头边缘的失真会导致面部拉伸，影响视觉吸引力。为此，我们提出了一种名为ImagePD的图像肖像矫正框架。该框架结合了变压器的长距离感知能力和扩散模型的多步去噪，实现了全局结构的鲁棒性和局部细节的优化。此外，考虑到获取视频标签的成本较高，我们通过时空扩散适配以及空间一致性与时序平滑的约束，将ImagePD应用于未标注的宽视角视频（称为VideoPD）。对于前者，我们鼓励去噪后的图像接近宽视角失真的分布模式以生成伪标签；对于后者，我们基于后向光学流推导矫正轨迹并进行平滑。与ImagePD相比，VideoPD在空间上保持高质量的面部矫正，并顺序减轻潜在的时间抖动。最后，为了建立评估基准并训练该框架，我们建立了一个包含大量人群数量、光照条件和背景多样性的视频肖像数据集。实验表明，所提出的方法在定量和定性上均优于现有解决方案，有助于生成具有稳定和自然肖像的高质量宽视角视频。代码和数据集将公开。 

---
# When Persuasion Overrides Truth in Multi-Agent LLM Debates: Introducing a Confidence-Weighted Persuasion Override Rate (CW-POR) 

**Title (ZH)**: 当说服力 overriding 事实真相在多代理大规模语言模型辩论中发生时：引入一种信心加权说服力 overriding 率（CW-POR） 

**Authors**: Mahak Agarwal, Divyam Khanna  

**Link**: [PDF](https://arxiv.org/pdf/2504.00374)  

**Abstract**: In many real-world scenarios, a single Large Language Model (LLM) may encounter contradictory claims-some accurate, others forcefully incorrect-and must judge which is true. We investigate this risk in a single-turn, multi-agent debate framework: one LLM-based agent provides a factual answer from TruthfulQA, another vigorously defends a falsehood, and the same LLM architecture serves as judge. We introduce the Confidence-Weighted Persuasion Override Rate (CW-POR), which captures not only how often the judge is deceived but also how strongly it believes the incorrect choice. Our experiments on five open-source LLMs (3B-14B parameters), where we systematically vary agent verbosity (30-300 words), reveal that even smaller models can craft persuasive arguments that override truthful answers-often with high confidence. These findings underscore the importance of robust calibration and adversarial testing to prevent LLMs from confidently endorsing misinformation. 

**Abstract (ZH)**: 在多种真实场景中，单一大型语言模型（LLM）可能遇到相互矛盾的断言——有些准确而另一些则是强制性的错误，并必须判断哪一个为真。我们在此研究单轮多代理辩论框架中的这一风险：一个基于LLM的代理提供一个事实性答案，另一个则极力辩护一个谬误，而相同的LLM架构担任裁判。我们引入了置信加权说服否决率（CW-POR），该指标不仅量化裁判被误导的频率，还衡量其对错误选择的信念强度。我们在五个开源LLM（3B-14B参数）上进行的实验中系统地变化代理的详尽程度（30-300词），揭示即使是较小的模型也能构建强有力的论据以超越真实的答案——并常常抱着很高的信心。这些发现强调了对LLM进行鲁棒校准和对抗性测试的重要性，以防止它们自信地推广谬误信息。 

---
# Hybrid Global-Local Representation with Augmented Spatial Guidance for Zero-Shot Referring Image Segmentation 

**Title (ZH)**: 全局-局部混合表示增强空间引导在零-shot 参考图像分割中的应用 

**Authors**: Ting Liu, Siyuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.00356)  

**Abstract**: Recent advances in zero-shot referring image segmentation (RIS), driven by models such as the Segment Anything Model (SAM) and CLIP, have made substantial progress in aligning visual and textual information. Despite these successes, the extraction of precise and high-quality mask region representations remains a critical challenge, limiting the full potential of RIS tasks. In this paper, we introduce a training-free, hybrid global-local feature extraction approach that integrates detailed mask-specific features with contextual information from the surrounding area, enhancing mask region representation. To further strengthen alignment between mask regions and referring expressions, we propose a spatial guidance augmentation strategy that improves spatial coherence, which is essential for accurately localizing described areas. By incorporating multiple spatial cues, this approach facilitates more robust and precise referring segmentation. Extensive experiments on standard RIS benchmarks demonstrate that our method significantly outperforms existing zero-shot RIS models, achieving substantial performance gains. We believe our approach advances RIS tasks and establishes a versatile framework for region-text alignment, offering broader implications for cross-modal understanding and interaction. Code is available at this https URL . 

**Abstract (ZH)**: Recent advances in零样本图像分割（RIS），受Segment Anything Model（SAM）和CLIP等模型的驱动，已在视觉和文本信息对齐方面取得了显著进展。尽管取得了这些成功，精确和高质量的掩膜区域表示的提取仍然是一个关键挑战，限制了RIS任务的全部潜力。在本文中，我们提出了一种无需训练的混合全局-局部特征提取方法，该方法结合了详细的掩膜特定特征和周围区域的上下文信息，从而增强掩膜区域表示。为了进一步加强掩膜区域与引参照述表达之间的对齐，我们提出了一种空间指导增强策略，以提高空间连贯性，这是准确定位描述区域所必需的。通过结合多种空间线索，该方法有助于实现更稳健和精确的引参照述分割。在标准RIS基准上的广泛实验表明，我们的方法显著优于现有零样本RIS模型，实现了显著的性能提升。我们认为，我们的方法推进了RIS任务，并建立了适用于区域-文本对齐的多功能框架，提供了跨模态理解和交互的更广泛意义。代码可在以下链接获取：this https URL。 

---
# Integrated LLM-Based Intrusion Detection with Secure Slicing xApp for Securing O-RAN-Enabled Wireless Network Deployments 

**Title (ZH)**: 基于LLM的集成入侵检测与安全切片xApp相结合，以保障O-RAN使能的无线网络部署安全性 

**Authors**: Joshua Moore, Aly Sabri Abdalla, Prabesh Khanal, Vuk Marojevic  

**Link**: [PDF](https://arxiv.org/pdf/2504.00341)  

**Abstract**: The Open Radio Access Network (O-RAN) architecture is reshaping telecommunications by promoting openness, flexibility, and intelligent closed-loop optimization. By decoupling hardware and software and enabling multi-vendor deployments, O-RAN reduces costs, enhances performance, and allows rapid adaptation to new technologies. A key innovation is intelligent network slicing, which partitions networks into isolated slices tailored for specific use cases or quality of service requirements. The RAN Intelligent Controller further optimizes resource allocation, ensuring efficient utilization and improved service quality for user equipment (UEs). However, the modular and dynamic nature of O-RAN expands the threat surface, necessitating advanced security measures to maintain network integrity, confidentiality, and availability. Intrusion detection systems have become essential for identifying and mitigating attacks. This research explores using large language models (LLMs) to generate security recommendations based on the temporal traffic patterns of connected UEs. The paper introduces an LLM-driven intrusion detection framework and demonstrates its efficacy through experimental deployments, comparing non fine-tuned and fine-tuned models for task-specific accuracy. 

**Abstract (ZH)**: 基于大规模语言模型的O-RAN环境下的 intrusion detection 方法研究 

---
# VNJPTranslate: A comprehensive pipeline for Vietnamese-Japanese translation 

**Title (ZH)**: VNJPTranslate: 一个全面的越南语-日语翻译管道 

**Authors**: Hoang Hai Phan, Nguyen Duc Minh Vu, Nam Dang Phuong  

**Link**: [PDF](https://arxiv.org/pdf/2504.00339)  

**Abstract**: Neural Machine Translation (NMT) driven by Transformer architectures has advanced significantly, yet faces challenges with low-resource language pairs like Vietnamese-Japanese (Vi-Ja). Issues include sparse parallel data and handling linguistic/cultural nuances. Recent progress in Large Language Models (LLMs) with strong reasoning, often refined via Reinforcement Learning (RL), enables high-quality synthetic data generation. We introduce VNJPTranslate, a pipeline designed to systematically address the Vi-Ja translation task. It features a targeted data augmentation strategy using advanced LLMs with Chain-of-Thought prompting for challenging segments identified via corpus analysis. Subsequently, we employ efficient fine-tuning techniques (Unsloth with QLoRA) on a capable, low-parameter autoregressive model (specifically, a fine-tuned version of the 1.8B parameter Sailor model, which is based on the Qwen architecture) to create a practical and high-performing translation system. This integrated approach aims to improve Vi-Ja translation quality significantly over existing baselines. 

**Abstract (ZH)**: 基于Transformer架构的神经机器翻译（NMT）在低资源语言对（如越南语-日语）翻译任务（Vi-Ja）中获得了显著进步，但仍面临并行数据稀疏和处理语言/文化差异的挑战。大规模语言模型（LLMs）借助强烈的推理能力，并通过强化学习（RL）进行优化，能够生成高质量的合成数据。我们提出VNJPTranslate流水线，旨在系统地解决越南语-日语翻译任务。该流水线采用先进LLMs的链式思考提示策略进行有针对性的数据增强，并通过语料库分析识别具有挑战性的段落。随后，我们采用高效的微调技术（UnSloth与QLoRA）对一个具有强大自回归能力但参数较少的模型（具体而言，是基于Qwen架构、已微调的1.8B参数Sailor模型）进行微调，以创建一个实用且高性能的翻译系统。该综合方法旨在显著提高越南语-日语翻译质量，超越现有基线。 

---
# Agentic Multimodal AI for Hyperpersonalized B2B and B2C Advertising in Competitive Markets: An AI-Driven Competitive Advertising Framework 

**Title (ZH)**: 代理型多模态AI在竞争市场中的超个性化B2B和B2C广告：一种AI驱动的竞争广告框架 

**Authors**: Sakhinana Sagar Srinivas, Akash Das, Shivam Gupta, Venkataramana Runkana  

**Link**: [PDF](https://arxiv.org/pdf/2504.00338)  

**Abstract**: The growing use of foundation models (FMs) in real-world applications demands adaptive, reliable, and efficient strategies for dynamic markets. In the chemical industry, AI-discovered materials drive innovation, but commercial success hinges on market adoption, requiring FM-driven advertising frameworks that operate in-the-wild. We present a multilingual, multimodal AI framework for autonomous, hyper-personalized advertising in B2B and B2C markets. By integrating retrieval-augmented generation (RAG), multimodal reasoning, and adaptive persona-based targeting, our system generates culturally relevant, market-aware ads tailored to shifting consumer behaviors and competition. Validation combines real-world product experiments with a Simulated Humanistic Colony of Agents to model consumer personas, optimize strategies at scale, and ensure privacy compliance. Synthetic experiments mirror real-world scenarios, enabling cost-effective testing of ad strategies without risky A/B tests. Combining structured retrieval-augmented reasoning with in-context learning (ICL), the framework boosts engagement, prevents market cannibalization, and maximizes ROAS. This work bridges AI-driven innovation and market adoption, advancing multimodal FM deployment for high-stakes decision-making in commercial marketing. 

**Abstract (ZH)**: 基础模型在实际应用中的日益增长促使我们需要适应性强、可靠且高效的动态市场策略。在化工行业中，AI发现的材料推动创新，但商业成功取决于市场接受度，因此需要由基础模型驱动的能够在野操作的广告框架。我们提出了一种跨语言、跨模态的AI框架，用于B2B和B2C市场的自主超个性化广告。通过整合检索增强生成（RAG）、跨模态推理和自适应角色定向，我们的系统生成了与文化相关的、市场意识强的广告，这些广告能够适应不断变化的消费者行为和竞争态势。验证结合了现实世界的产品实验与仿真人文型代理群体，用于建模消费者角色、大规模优化策略并确保隐私合规。合成实验镜像现实世界场景，能够在不进行风险较高的A/B测试的情况下，经济有效地测试广告策略。结合结构化检索增强推理与上下文学习（ICL），该框架增强了互动性，防止市场竞争侵蚀，并最大化了ROAS。这项工作将AI驱动的创新与市场接纳结合起来，推进了跨模态基础模型在商业营销中高风险决策中的应用部署。 

---
# SeizureTransformer: Scaling U-Net with Transformer for Simultaneous Time-Step Level Seizure Detection from Long EEG Recordings 

**Title (ZH)**: SeizureTransformer: 通过Transformer扩展U-Net以实现长EEG记录的同步时间步长 seizures 检测 

**Authors**: Kerui Wu, Ziyue Zhao, Bülent Yener  

**Link**: [PDF](https://arxiv.org/pdf/2504.00336)  

**Abstract**: Epilepsy is a common neurological disorder that affects around 65 million people worldwide. Detecting seizures quickly and accurately is vital, given the prevalence and severity of the associated complications. Recently, deep learning-based automated seizure detection methods have emerged as solutions; however, most existing methods require extensive post-processing and do not effectively handle the crucial long-range patterns in EEG data. In this work, we propose SeizureTransformer, a simple model comprised of (i) a deep encoder comprising 1D convolutions (ii) a residual CNN stack and a transformer encoder to embed previous output into high-level representation with contextual information, and (iii) streamlined decoder which converts these features into a sequence of probabilities, directly indicating the presence or absence of seizures at every time step. Extensive experiments on public and private EEG seizure detection datasets demonstrate that our model significantly outperforms existing approaches (ranked in the first place in the 2025 "seizure detection challenge" organized in the International Conference on Artificial Intelligence in Epilepsy and Other Neurological Disorders), underscoring its potential for real-time, precise seizure detection. 

**Abstract (ZH)**: 癫痫是一种影响全球约6500万人的常见神经 disorder，快速准确地检测癫痫发作至关重要。近年来，基于深度学习的自动化癫痫发作检测方法逐渐成为解决方案；然而，现有的大多数方法需要大量的后处理，并不能很好地处理 EEG 数据中的关键长范围模式。在本工作中，我们提出了 SeizureTransformer，这是一种简单的模型，包括（i）一个由1D卷积构成的深度编码器；（ii）一个残差 CNN 层叠和变压器编码器，用于将先前输出嵌入到包含上下文信息的高层表示；（iii）一个精简的解码器，将这些特征转化为时间步骤序列的概率，直接指示每个时间步骤是否存在癫痫发作。在公开和私有 EEG 癫痫发作检测数据集上的广泛实验表明，我们的模型在国际癫痫与其他神经疾病人工智能会议组织的2025年“癫痫发作检测挑战赛”中排名首位，突显了其在实时、精确的癫痫发作检测方面的潜力。 

---
# Detecting and Mitigating Bias in LLMs through Knowledge Graph-Augmented Training 

**Title (ZH)**: 通过知识图谱增强训练检测和缓解LLM中的偏见 

**Authors**: Rajeev Kumar, Harishankar Kumar, Kumari Shalini  

**Link**: [PDF](https://arxiv.org/pdf/2504.00310)  

**Abstract**: Large language models have revolutionized natural language processing with their surprising capability to understand and generate human-like text. However, many of these models inherit and further amplify the biases present in their training data, raising ethical and fairness concerns. The detection and mitigation of such biases are vital to ensuring that LLMs act responsibly and equitably across diverse domains. This work investigates Knowledge Graph-Augmented Training (KGAT) as a novel method to mitigate bias in LLM. Using structured domain-specific knowledge from real-world knowledge graphs, we improve the understanding of the model and reduce biased output. Public datasets for bias assessment include Gender Shades, Bias in Bios, and FairFace, while metrics such as demographic parity and equal opportunity facilitate rigorous detection. We also performed targeted mitigation strategies to correct biased associations, leading to a significant drop in biased output and improved bias metrics. Equipped with real-world datasets and knowledge graphs, our framework is both scalable and effective, paving the way toward responsible deployment in sensitive and high-stakes applications. 

**Abstract (ZH)**: 大型语言模型通过其令人惊讶的语言理解与生成人类文本的能力， revolutionized 自然语言处理。然而，这些模型继承并进一步放大了训练数据中存在的偏见，引发了伦理与公平性方面的关注。检测与缓解这些偏见对于确保大型语言模型在多领域中负责任且公平地行动至关重要。本文探讨了知识图谱增强训练（KGAT）作为缓解大型语言模型偏见的新方法。通过使用现实世界知识图谱中的结构化领域特定知识，我们改进了模型的理解并减少了有偏见的输出。用于偏见评估的公开数据集包括 Gender Shades、Bias in Bios 和 FairFace，而人口统计平等等指标促使我们进行严格的检测。我们还实施了有针对性的缓解策略以纠正有偏见的关联，显著减少了有偏见的输出并改善了偏见指标。配备了现实世界数据集和知识图谱，我们的框架既具扩展性又有效，为在敏感和高风险应用中负责任地部署奠定了基础。 

---
# FedPaI: Achieving Extreme Sparsity in Federated Learning via Pruning at Initialization 

**Title (ZH)**: FedPaI: 在初始化时剪枝以实现联邦学习中的极端稀疏性 

**Authors**: Haonan Wang, Zeli Liu, Kajimusugura Hoshino, Tuo Zhang, John Paul Walters, Stephen Crago  

**Link**: [PDF](https://arxiv.org/pdf/2504.00308)  

**Abstract**: Federated Learning (FL) enables distributed training on edge devices but faces significant challenges due to resource constraints in edge environments, impacting both communication and computational efficiency. Existing iterative pruning techniques improve communication efficiency but are limited by their centralized design, which struggles with FL's decentralized and data-imbalanced nature, resulting in suboptimal sparsity levels. To address these issues, we propose FedPaI, a novel efficient FL framework that leverages Pruning at Initialization (PaI) to achieve extreme sparsity. FedPaI identifies optimal sparse connections at an early stage, maximizing model capacity and significantly reducing communication and computation overhead by fixing sparsity patterns at the start of training. To adapt to diverse hardware and software environments, FedPaI supports both structured and unstructured pruning. Additionally, we introduce personalized client-side pruning mechanisms for improved learning capacity and sparsity-aware server-side aggregation for enhanced efficiency. Experimental results demonstrate that FedPaI consistently outperforms existing efficient FL that applies conventional iterative pruning with significant leading in efficiency and model accuracy. For the first time, our proposed FedPaI achieves an extreme sparsity level of up to 98% without compromising the model accuracy compared to unpruned baselines, even under challenging non-IID settings. By employing our FedPaI with joint optimization of model learning capacity and sparsity, FL applications can benefit from faster convergence and accelerate the training by 6.4 to 7.9 times. 

**Abstract (ZH)**: Federated Learning中基于初始化剪枝的高效框架FedPaI：实现极端稀疏性的同时保持模型准确性 

---
# Inference-Time Scaling for Complex Tasks: Where We Stand and What Lies Ahead 

**Title (ZH)**: 复杂任务下的推理时伸缩性：当前状况与未来展望 

**Authors**: Vidhisha Balachandran, Jingya Chen, Lingjiao Chen, Shivam Garg, Neel Joshi, Yash Lara, John Langford, Besmira Nushi, Vibhav Vineet, Yue Wu, Safoora Yousefi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00294)  

**Abstract**: Inference-time scaling can enhance the reasoning capabilities of large language models (LLMs) on complex problems that benefit from step-by-step problem solving. Although lengthening generated scratchpads has proven effective for mathematical tasks, the broader impact of this approach on other tasks remains less clear. In this work, we investigate the benefits and limitations of scaling methods across nine state-of-the-art models and eight challenging tasks, including math and STEM reasoning, calendar planning, NP-hard problems, navigation, and spatial reasoning. We compare conventional models (e.g., GPT-4o) with models fine-tuned for inference-time scaling (e.g., o1) through evaluation protocols that involve repeated model calls, either independently or sequentially with feedback. These evaluations approximate lower and upper performance bounds and potential for future performance improvements for each model, whether through enhanced training or multi-model inference systems. Our extensive empirical analysis reveals that the advantages of inference-time scaling vary across tasks and diminish as problem complexity increases. In addition, simply using more tokens does not necessarily translate to higher accuracy in these challenging regimes. Results from multiple independent runs with conventional models using perfect verifiers show that, for some tasks, these models can achieve performance close to the average performance of today's most advanced reasoning models. However, for other tasks, a significant performance gap remains, even in very high scaling regimes. Encouragingly, all models demonstrate significant gains when inference is further scaled with perfect verifiers or strong feedback, suggesting ample potential for future improvements. 

**Abstract (ZH)**: 推理时的扩展可以增强大型语言模型在复杂问题上的推理能力，这些问题是步骤式问题解决所获益的。虽然延长生成草稿区片段对数学任务的有效性已经得到了证明，但这种方法对其他任务的更广泛影响仍不明确。在本工作中，我们调查了九个最先进的模型和八个具有挑战性的任务（包括数学和STEM推理、日历规划、NP难问题、导航和空间推理）中扩展方法的好处和限制。我们将传统模型（如GPT-4o）与针对推理时扩展进行微调的模型（如o1）进行了比较，评估协议包括重复调用模型，要么独立地要么在反馈情况下顺序调用。这些评估接近每个模型的下界和上界性能，并且反馈可能对未来性能改进的潜力提供估算。广泛的实证分析显示，推理时扩展的优势随着任务复杂度的增加而变化并在复杂问题上减弱。此外，在这些具有挑战性的环境中，简单地使用更多标记并不必然意味着更高的准确性。使用完美验证器对传统模型进行多次独立运行的评估表明，对于某些任务，这些模型可以达到接近当今最先进的推理模型平均性能的水平。然而，对于其他任务，在非常高的扩展范围内仍然存在显著的性能差距。令人鼓舞的是，所有模型在使用完美验证器或强反馈进一步扩展推理时都显示出显著的性能提升，这表明未来改进的潜力巨大。 

---
# Do Chinese models speak Chinese languages? 

**Title (ZH)**: 中国人使用的模型说中文吗？ 

**Authors**: Andrea W Wen-Yi, Unso Eun Seo Jo, David Mimno  

**Link**: [PDF](https://arxiv.org/pdf/2504.00289)  

**Abstract**: The release of top-performing open-weight LLMs has cemented China's role as a leading force in AI development. Do these models support languages spoken in China? Or do they speak the same languages as Western models? Comparing multilingual capabilities is important for two reasons. First, language ability provides insights into pre-training data curation, and thus into resource allocation and development priorities. Second, China has a long history of explicit language policy, varying between inclusivity of minority languages and a Mandarin-first policy. To test whether Chinese LLMs today reflect an agenda about China's languages, we test performance of Chinese and Western open-source LLMs on Asian regional and Chinese minority languages. Our experiments on Information Parity and reading comprehension show Chinese models' performance across these languages correlates strongly (r=0.93) with Western models', with the sole exception being better Mandarin. Sometimes, Chinese models cannot identify languages spoken by Chinese minorities such as Kazakh and Uyghur, even though they are good at French and German. These results provide a window into current development priorities, suggest options for future development, and indicate guidance for end users. 

**Abstract (ZH)**: 顶级开放权重LLM的发布已确立了中国在AI发展中的领导地位。这些模型支持中国的语言吗？还是说它们使用与西方模型相同的语言？比较多语言能力对于两个原因非常重要。首先，语言能力提供了有关预训练数据收集的见解，从而反映了资源分配和开发优先级。其次，中国有着悠久的语言政策历史，政策之间在少数语言包容性和汉语优先政策之间存在差异。为了测试当前的Chinese LLMs是否反映了关于中国语言的议程，我们测试了Chinese和Western开源LLM在亚洲区域和中国少数民族语言上的性能。我们的Information Parity和阅读理解实验结果显示，Chinese模型在这几种语言上的性能与Western模型高度相关（r=0.93），唯一例外是在普通话上稍好。有时，Chinese模型无法识别如哈萨克语和乌兹别克语等中国少数民族的语言，尽管它们在法语和德语上表现很好。这些结果提供了一个了解当前开发优先级的窗口，提出了未来发展的选项，并指出了终端用户的应用指导。 

---
# Digital Twins in Biopharmaceutical Manufacturing: Review and Perspective on Human-Machine Collaborative Intelligence 

**Title (ZH)**: 生物制药制造中的数字孪生：人类-机器协作智能的回顾与展望 

**Authors**: Mohammed Aatif Shahab, Francesco Destro, Richard D. Braatz  

**Link**: [PDF](https://arxiv.org/pdf/2504.00286)  

**Abstract**: The biopharmaceutical industry is increasingly developing digital twins to digitalize and automate the manufacturing process in response to the growing market demands. However, this shift presents significant challenges for human operators, as the complexity and volume of information can overwhelm their ability to manage the process effectively. These issues are compounded when digital twins are designed without considering interaction and collaboration with operators, who are responsible for monitoring processes and assessing situations, particularly during abnormalities. Our review of current trends in biopharma digital twin development reveals a predominant focus on technology and often overlooks the critical role of human operators. To bridge this gap, this article proposes a collaborative intelligence framework that emphasizes the integration of operators with digital twins. Approaches to system design that can enhance operator trust and human-machine interface usability are presented. Moreover, innovative training programs for preparing operators to understand and utilize digital twins are discussed. The framework outlined in this article aims to enhance collaboration between operators and digital twins effectively by using their full capabilities to boost resilience and productivity in biopharmaceutical manufacturing. 

**Abstract (ZH)**: 生物制药行业正在越来越多地开发数字孪生以实现制造过程的数字化和自动化，以应对市场需求的增长。然而，这一转变对人类操作人员提出了重大挑战，因为信息的复杂性和数量可能超出他们有效管理过程的能力。当数字孪生的设计忽视了与操作人员的互动和协作时，这些问题会进一步加剧，尤其是操作人员在异常情况下的监控和评估工作。我们对生物制药数字孪生发展现状的回顾表明，当前主要集中在技术上，往往忽视了操作人员的关键作用。为解决这一差距，本文提出了一种协同智能框架，强调将操作人员与数字孪生集成。本文提出了增强操作人员信任和人机界面易用性的系统设计方法，并讨论了创新的操作人员培训计划，以便他们能够理解并利用数字孪生。本文概述的框架旨在通过充分利用操作人员和数字孪生的能力，有效提升生物制药制造的韧性和生产效率。 

---
# SciReplicate-Bench: Benchmarking LLMs in Agent-driven Algorithmic Reproduction from Research Papers 

**Title (ZH)**: SciReplicate-Bench：基于代理驱动算法再现从研究论文中评估LLMs的效果 

**Authors**: Yanzheng Xiang, Hanqi Yan, Shuyin Ouyang, Lin Gui, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2504.00255)  

**Abstract**: This study evaluates large language models (LLMs) in generating code from algorithm descriptions from recent NLP papers. The task requires two key competencies: (1) algorithm comprehension: synthesizing information from papers and academic literature to understand implementation logic, and (2) coding expertise: identifying dependencies and correctly implementing necessary APIs. To facilitate rigorous evaluation, we introduce SciReplicate-Bench, a benchmark of 100 tasks from 36 NLP papers published in 2024, featuring detailed annotations and comprehensive test cases. Building on SciReplicate-Bench, we propose Sci-Reproducer, a multi-agent framework consisting of a Paper Agent that interprets algorithmic concepts from literature and a Code Agent that retrieves dependencies from repositories and implement solutions. To assess algorithm understanding, we introduce reasoning graph accuracy, which quantifies similarity between generated and reference reasoning graphs derived from code comments and structure. For evaluating implementation quality, we employ execution accuracy, CodeBLEU, and repository dependency/API recall metrics. In our experiments, we evaluate various powerful Non-Reasoning LLMs and Reasoning LLMs as foundational models. The best-performing LLM using Sci-Reproducer achieves only 39% execution accuracy, highlighting the benchmark's this http URL analysis identifies missing or inconsistent algorithm descriptions as key barriers to successful reproduction. We will open-source our benchmark, and code at this https URL. 

**Abstract (ZH)**: 本研究评估了大型语言模型在从近期NLP论文中的算法描述生成代码方面的性能。该任务要求具备两个关键能力：（1）算法理解：从论文和学术文献中综合信息以理解实现逻辑，（2）编程技能：识别依赖关系并正确实现必要的API。为确保严格的评估，我们引入了SciReplicate-Bench这一基准，包括来自2024年36篇NLP论文的100个任务，附有详细注释和全面的测试案例。基于SciReplicate-Bench，我们提出了一种多代理框架Sci-Reproducer，其中包括一个论文代理，负责从文献中解释算法概念，以及一个代码代理，负责从代码库中检索依赖关系并实现解决方案。为了评估算法理解，我们引入了推理图准确性，该指标量化了生成的推理图与来自代码注释和结构的参考推理图之间的相似性。为了评估实现质量，我们采用了执行准确性、CodeBLEU以及代码库依赖关系/API召回率指标。在实验中，我们评估了多种强大的非推理型LLM和推理型LLM作为基础模型。使用Sci-Reproducer的性能最佳模型仅实现39%的执行准确性，突显了基准的挑战性。分析指出，缺失或不一致的算法描述是成功复现的主要障碍。我们将在GitHub开源我们的基准和代码。 

---
# ElaLoRA: Elastic & Learnable Low-Rank Adaptation for Efficient Model Fine-Tuning 

**Title (ZH)**: ElaLoRA: 弹性可学习低秩适应以实现高效的模型微调 

**Authors**: Huandong Chang, Zicheng Ma, Mingyuan Ma, Zhenting Qi, Andrew Sabot, Hong Jiang, H. T. Kung  

**Link**: [PDF](https://arxiv.org/pdf/2504.00254)  

**Abstract**: Low-Rank Adaptation (LoRA) has become a widely adopted technique for fine-tuning large-scale pre-trained models with minimal parameter updates. However, existing methods rely on fixed ranks or focus solely on either rank pruning or expansion, failing to adapt ranks dynamically to match the importance of different layers during training. In this work, we propose ElaLoRA, an adaptive low-rank adaptation framework that dynamically prunes and expands ranks based on gradient-derived importance scores. To the best of our knowledge, ElaLoRA is the first method that enables both rank pruning and expansion during fine-tuning. Experiments across multiple benchmarks demonstrate that ElaLoRA consistently outperforms existing PEFT methods across different parameter budgets. Furthermore, our studies validate that layers receiving higher rank allocations contribute more significantly to model performance, providing theoretical justification for our adaptive strategy. By introducing a principled and adaptive rank allocation mechanism, ElaLoRA offers a scalable and efficient fine-tuning solution, particularly suited for resource-constrained environments. 

**Abstract (ZH)**: ElaLoRA: An Adaptive Low-Rank Adaptation Framework for Fine-Tuning Large-Scale Pre-Trained Models 

---
# MultiMorph: On-demand Atlas Construction 

**Title (ZH)**: MultiMorph: 按需成图方法 

**Authors**: S. Mazdak Abulnaga, Andrew Hoopes, Neel Dey, Malte Hoffmann, Marianne Rakic, Bruce Fischl, John Guttag, Adrian Dalca  

**Link**: [PDF](https://arxiv.org/pdf/2504.00247)  

**Abstract**: We present MultiMorph, a fast and efficient method for constructing anatomical atlases on the fly. Atlases capture the canonical structure of a collection of images and are essential for quantifying anatomical variability across populations. However, current atlas construction methods often require days to weeks of computation, thereby discouraging rapid experimentation. As a result, many scientific studies rely on suboptimal, precomputed atlases from mismatched populations, negatively impacting downstream analyses. MultiMorph addresses these challenges with a feedforward model that rapidly produces high-quality, population-specific atlases in a single forward pass for any 3D brain dataset, without any fine-tuning or optimization. MultiMorph is based on a linear group-interaction layer that aggregates and shares features within the group of input images. Further, by leveraging auxiliary synthetic data, MultiMorph generalizes to new imaging modalities and population groups at test-time. Experimentally, MultiMorph outperforms state-of-the-art optimization-based and learning-based atlas construction methods in both small and large population settings, with a 100-fold reduction in time. This makes MultiMorph an accessible framework for biomedical researchers without machine learning expertise, enabling rapid, high-quality atlas generation for diverse studies. 

**Abstract (ZH)**: MultiMorph: 一种快速高效的大脑结构图谱构建方法 

---
# Synthesizing Public Opinions with LLMs: Role Creation, Impacts, and the Future to eDemorcacy 

**Title (ZH)**: 使用大语言模型合成公众意见：角色创建、影响及对eDemocracy的未来展望 

**Authors**: Rabimba Karanjai, Boris Shor, Amanda Austin, Ryan Kennedy, Yang Lu, Lei Xu, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00241)  

**Abstract**: This paper investigates the use of Large Language Models (LLMs) to synthesize public opinion data, addressing challenges in traditional survey methods like declining response rates and non-response bias. We introduce a novel technique: role creation based on knowledge injection, a form of in-context learning that leverages RAG and specified personality profiles from the HEXACO model and demographic information, and uses that for dynamically generated prompts. This method allows LLMs to simulate diverse opinions more accurately than existing prompt engineering approaches. We compare our results with pre-trained models with standard few-shot prompts. Experiments using questions from the Cooperative Election Study (CES) demonstrate that our role-creation approach significantly improves the alignment of LLM-generated opinions with real-world human survey responses, increasing answer adherence. In addition, we discuss challenges, limitations and future research directions. 

**Abstract (ZH)**: 本文探讨了使用大型语言模型（LLMs）合成公众意见数据，解决传统调查方法中如响应率下降和无响应偏差等挑战。我们提出了一种新颖的技术：基于知识注入的角色创建，这是一种利用RAG和HEXACO模型指定的人格特征以及人口统计信息的上下文学习方式，并用于动态生成提示。该方法使LLMs能够比现有的提示工程方法更准确地模拟多样化的观点。我们还将我们的结果与使用标准少样本提示的预训练模型进行了比较。使用合作选举研究（CES）的问题进行的实验表明，我们的角色创建方法显著提高了LLM生成观点与实际人类调查响应的一致性，增加了答案的符合度。此外，我们讨论了挑战、局限性和未来的研究方向。 

---
# GazeLLM: Multimodal LLMs incorporating Human Visual Attention 

**Title (ZH)**: GazeLLM: 结合人类视觉注意力的多模态大模型 

**Authors**: Jun Rekimoto  

**Link**: [PDF](https://arxiv.org/pdf/2504.00221)  

**Abstract**: Large Language Models (LLMs) are advancing into Multimodal LLMs (MLLMs), capable of processing image, audio, and video as well as text. Combining first-person video, MLLMs show promising potential for understanding human activities through video and audio, enabling many human-computer interaction and human-augmentation applications such as human activity support, real-world agents, and skill transfer to robots or other individuals. However, handling high-resolution, long-duration videos generates large latent representations, leading to substantial memory and processing demands, limiting the length and resolution MLLMs can manage. Reducing video resolution can lower memory usage but often compromises comprehension. This paper introduces a method that optimizes first-person video analysis by integrating eye-tracking data, and proposes a method that decomposes first-person vision video into sub areas for regions of gaze focus. By processing these selectively gazed-focused inputs, our approach achieves task comprehension equivalent to or even better than processing the entire image at full resolution, but with significantly reduced video data input (reduce the number of pixels to one-tenth), offering an efficient solution for using MLLMs to interpret and utilize human skills. 

**Abstract (ZH)**: 大型语言模型正进而发展为多模态大语言模型，能够处理图像、音频和视频以及文本。结合第一人称视角视频，多模态大语言模型展示了通过视频和音频理解人类活动的潜力，这使得许多人类-计算机交互和人类增强应用成为可能，例如人类活动支持、真实世界代理和技能传递给机器人或其他个体。然而，处理高分辨率、长时间的视频会生成大量潜在表示，导致显著的内存和处理需求，限制了多模态大语言模型能够管理的视频长度和分辨率。降低视频分辨率可以减少内存使用，但往往会牺牲理解能力。本文提出了一种通过整合眼动数据优化第一人称视角视频分析的方法，并提出了一种将第一人称视角视频分解为关注区域的方法。通过处理这些选择性关注输入，我们的方法在不降低任务理解能力的前提下实现了全分辨率图像的处理效果，但视频数据输入大幅减少（像素数量减少至十分之一），提供了一个高效利用多模态大语言模型解释和应用人类技能的解决方案。 

---
# Can Diffusion Models Disentangle? A Theoretical Perspective 

**Title (ZH)**: 扩散模型能否解耦？一个理论视角 

**Authors**: Liming Wang, Muhammad Jehanzeb Mirza, Yishu Gong, Yuan Gong, Jiaqi Zhang, Brian H. Tracey, Katerina Placek, Marco Vilela, James R. Glass  

**Link**: [PDF](https://arxiv.org/pdf/2504.00220)  

**Abstract**: This paper presents a novel theoretical framework for understanding how diffusion models can learn disentangled representations. Within this framework, we establish identifiability conditions for general disentangled latent variable models, analyze training dynamics, and derive sample complexity bounds for disentangled latent subspace models. To validate our theory, we conduct disentanglement experiments across diverse tasks and modalities, including subspace recovery in latent subspace Gaussian mixture models, image colorization, image denoising, and voice conversion for speech classification. Additionally, our experiments show that training strategies inspired by our theory, such as style guidance regularization, consistently enhance disentanglement performance. 

**Abstract (ZH)**: 本文提出了一种新的理论框架，用于理解扩散模型如何学习解耦表示。在此框架内，我们建立了通用解耦潜在变量模型的可识别性条件，分析了训练动力学，并推导出了解耦潜在子空间模型的样本复杂性边界。为了验证我们的理论，我们在多任务和多种模态下进行了解耦实验，包括潜在子空间高斯混合模型中的子空间恢复、图像颜色化、图像去噪以及用于语音分类的语音转换。此外，我们的实验表明，受我们的理论启发的训练策略，如风格引导正则化，始终能提升解耦性能。 

---
# $\textit{Agents Under Siege}$: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks 

**Title (ZH)**: 《四面受敌的智能体》：通过优化提示攻击打破实用多智能体LLM系统 

**Authors**: Rana Muhammad Shahroz Khan, Zhen Tan, Sukwon Yun, Charles Flemming, Tianlong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.00218)  

**Abstract**: Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms. 

**Abstract (ZH)**: 大多数关于大型语言模型（LLM）安全性的讨论集中在单代理设置上，但多代理LLM系统现在创造了新的对抗风险，因为其行为依赖于代理之间的通信和分散式推理。在此项工作中，我们创新性地将攻击目标集中在具有有限令牌带宽、消息传递延迟等约束的实用系统上，并设计了一种具有传递不变性对抗攻击，旨在优化跨延迟和带宽约束网络拓扑的提示分布，以绕过系统内的分布式安全机制。我们将攻击路径表述为最大流最小成本问题，并结合新型传递不变性规避损失（PIEL），利用图基优化方法在最大化攻击成功率的同时最小化检测风险。在包括Llama、Mistral、Gemma、DeepSeek及其他变体的多种模型上，在JailBreakBench和AdversarialBench等不同数据集上评估，我们的方法比传统攻击高出高达7倍的效果，揭示了多代理系统中的关键漏洞。此外，我们证明现有的防御措施，包括Llama-Guard和PromptGuard的变体，无法阻止我们的攻击，强调了针对多代理系统的特定安全机制的迫切需求。 

---
# RailGoerl24: Görlitz Rail Test Center CV Dataset 2024 

**Title (ZH)**: RailGoerl24: Görlitz 铁路测试中心 CV 数据集 2024 

**Authors**: Rustam Tagiew, Ilkay Wunderlich, Mark Sastuba, Steffen Seitz  

**Link**: [PDF](https://arxiv.org/pdf/2504.00204)  

**Abstract**: Driverless train operation for open tracks on urban guided transport and mainline railways requires, among other things automatic detection of actual and potential obstacles, especially humans, in the danger zone of the train's path. Machine learning algorithms have proven to be powerful state-of-the-art tools for this task. However, these algorithms require large amounts of high-quality annotated data containing human beings in railway-specific environments as training data. Unfortunately, the amount of publicly available datasets is not yet sufficient and is significantly inferior to the datasets in the road domain. Therefore, this paper presents RailGoerl24, an on-board visual light Full HD camera dataset of 12205 frames recorded in a railway test center of TÜV SÜD Rail, in Görlitz, Germany. Its main purpose is to support the development of driverless train operation for guided transport. RailGoerl24 also includes a terrestrial LiDAR scan covering parts of the area used to acquire the RGB data. In addition to the raw data, the dataset contains 33556 boxwise annotations in total for the object class 'person'. The faces of recorded actors are not blurred or altered in any other way. RailGoerl24, soon available at this http URL, can also be used for tasks beyond collision prediction. 

**Abstract (ZH)**: 基于轨道的无人驾驶列车运营视觉数据集RailGoerl24及其应用 

---
# Identifying Sparsely Active Circuits Through Local Loss Landscape Decomposition 

**Title (ZH)**: 通过局部损失景观分解识别稀疏激活电路 

**Authors**: Brianna Chrisman, Lucius Bushnaq, Lee Sharkey  

**Link**: [PDF](https://arxiv.org/pdf/2504.00194)  

**Abstract**: Much of mechanistic interpretability has focused on understanding the activation spaces of large neural networks. However, activation space-based approaches reveal little about the underlying circuitry used to compute features. To better understand the circuits employed by models, we introduce a new decomposition method called Local Loss Landscape Decomposition (L3D). L3D identifies a set of low-rank subnetworks: directions in parameter space of which a subset can reconstruct the gradient of the loss between any sample's output and a reference output vector. We design a series of progressively more challenging toy models with well-defined subnetworks and show that L3D can nearly perfectly recover the associated subnetworks. Additionally, we investigate the extent to which perturbing the model in the direction of a given subnetwork affects only the relevant subset of samples. Finally, we apply L3D to a real-world transformer model and a convolutional neural network, demonstrating its potential to identify interpretable and relevant circuits in parameter space. 

**Abstract (ZH)**: 基于局部损失景观分解的方法揭示神经网络模型中使用的电路结构 

---
# Are Domain Generalization Benchmarks with Accuracy on the Line Misspecified? 

**Title (ZH)**: 泛化基准的准确度是否失准？ 

**Authors**: Olawale Salaudeen, Nicole Chiou, Shiny Weng, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2504.00186)  

**Abstract**: Spurious correlations are unstable statistical associations that hinder robust decision-making. Conventional wisdom suggests that models relying on such correlations will fail to generalize out-of-distribution (OOD), especially under strong distribution shifts. However, empirical evidence challenges this view as naive in-distribution empirical risk minimizers often achieve the best OOD accuracy across popular OOD generalization benchmarks. In light of these results, we propose a different perspective: many widely used benchmarks for evaluating robustness to spurious correlations are misspecified. Specifically, they fail to include shifts in spurious correlations that meaningfully impact OOD generalization, making them unsuitable for evaluating the benefit of removing such correlations. We establish conditions under which a distribution shift can reliably assess a model's reliance on spurious correlations. Crucially, under these conditions, we should not observe a strong positive correlation between in-distribution and OOD accuracy, often called "accuracy on the line." Yet, most state-of-the-art benchmarks exhibit this pattern, suggesting they do not effectively assess robustness. Our findings expose a key limitation in current benchmarks used to evaluate domain generalization algorithms, that is, models designed to avoid spurious correlations. We highlight the need to rethink how robustness to spurious correlations is assessed, identify well-specified benchmarks the field should prioritize, and enumerate strategies for designing future benchmarks that meaningfully reflect robustness under distribution shift. 

**Abstract (ZH)**: 虚假相关性是不稳定的统计关联，妨碍了稳健决策的制定。传统智慧认为依赖此类关联的模型在出分布（OOD）外泛化能力较差，特别是在分布强烈变化时。然而，实证证据挑战了这一观点，因为在分布内的经验风险最小化方法往往能达到最佳的OOD准确性。鉴于这些结果，我们提出了一种不同的视角：许多用于评估对虚假相关性稳健性的常用基准可能是不恰当的。具体而言，它们未能包含对OOD泛化有实质性影响的虚假相关性的变化，使得它们不适合评估去除这些虚假相关性的益处。我们确立了在哪些条件下分布变化可以可靠地评估模型对虚假相关性的依赖性。至关重要的是，在这些条件下，我们不应该观察到在分布内和OOD准确性之间存在强烈的正相关，即所谓的“准确线性”。然而，大多数最先进的基准都表现出这种模式，表明它们未能有效评估稳健性。我们的研究揭示了当前用于评估领域泛化算法的基准的一个关键局限性，即设计为避免虚假相关性的模型。我们强调需要重新思考如何评估对虚假相关性的稳健性，指出了领域应优先考虑的恰当基准，并列举了设计未来的基准以更好地反映分布变化下的稳健性的策略。 

---
# Contradiction Detection in RAG Systems: Evaluating LLMs as Context Validators for Improved Information Consistency 

**Title (ZH)**: RAG系统中的矛盾检测：评估LLM作为上下文验证器以提高信息一致性 

**Authors**: Vignesh Gokul, Srikanth Tenneti, Alwarappan Nakkiran  

**Link**: [PDF](https://arxiv.org/pdf/2504.00180)  

**Abstract**: Retrieval Augmented Generation (RAG) systems have emerged as a powerful method for enhancing large language models (LLMs) with up-to-date information. However, the retrieval step in RAG can sometimes surface documents containing contradictory information, particularly in rapidly evolving domains such as news. These contradictions can significantly impact the performance of LLMs, leading to inconsistent or erroneous outputs. This study addresses this critical challenge in two ways. First, we present a novel data generation framework to simulate different types of contradictions that may occur in the retrieval stage of a RAG system. Second, we evaluate the robustness of different LLMs in performing as context validators, assessing their ability to detect contradictory information within retrieved document sets. Our experimental results reveal that context validation remains a challenging task even for state-of-the-art LLMs, with performance varying significantly across different types of contradictions. While larger models generally perform better at contradiction detection, the effectiveness of different prompting strategies varies across tasks and model architectures. We find that chain-of-thought prompting shows notable improvements for some models but may hinder performance in others, highlighting the complexity of the task and the need for more robust approaches to context validation in RAG systems. 

**Abstract (ZH)**: 基于检索增强生成（RAG）系统的矛盾信息缓解研究 

---
# Boundless Byte Pair Encoding: Breaking the Pre-tokenization Barrier 

**Title (ZH)**: 无界字对编码：突破预分词障碍 

**Authors**: Craig W. Schmidt, Varshini Reddy, Chris Tanner, Yuval Pinter  

**Link**: [PDF](https://arxiv.org/pdf/2504.00178)  

**Abstract**: Pre-tokenization, the initial step in many modern tokenization pipelines, segments text into smaller units called pretokens, typically splitting on whitespace and punctuation. While this process encourages having full, individual words as tokens, it introduces a fundamental limitation in most tokenization algorithms such as Byte Pair Encoding (BPE). Specifically, pre-tokenization causes the distribution of tokens in a corpus to heavily skew towards common, full-length words. This skewed distribution limits the benefits of expanding to larger vocabularies, since the additional tokens appear with progressively lower counts. To overcome this barrier, we propose BoundlessBPE, a modified BPE algorithm that relaxes the pretoken boundary constraint. Our approach selectively merges two complete pretokens into a larger unit we term a superword. Superwords are not necessarily semantically cohesive. For example, the pretokens " of" and " the" might be combined to form the superword " of the". This merging strategy results in a substantially more uniform distribution of tokens across a corpus than standard BPE, and compresses text more effectively, with an approximate 20% increase in bytes per token. 

**Abstract (ZH)**: 无界BPE：一种放松预词化边界约束的BPE算法 

---
# MetaCLBench: Meta Continual Learning Benchmark on Resource-Constrained Edge Devices 

**Title (ZH)**: MetaCLBench: 有限资源边缘设备上的元连续学习基准 

**Authors**: Sijia Li, Young D. Kwon, Lik-Hang Lee, Pan Hui  

**Link**: [PDF](https://arxiv.org/pdf/2504.00174)  

**Abstract**: Meta-Continual Learning (Meta-CL) has emerged as a promising approach to minimize manual labeling efforts and system resource requirements by enabling Continual Learning (CL) with limited labeled samples. However, while existing methods have shown success in image-based tasks, their effectiveness remains unexplored for sequential time-series data from sensor systems, particularly audio inputs. To address this gap, we conduct a comprehensive benchmark study evaluating six representative Meta-CL approaches using three network architectures on five datasets from both image and audio modalities. We develop MetaCLBench, an end-to-end Meta-CL benchmark framework for edge devices to evaluate system overheads and investigate trade-offs among performance, computational costs, and memory requirements across various Meta-CL methods. Our results reveal that while many Meta-CL methods enable to learn new classes for both image and audio modalities, they impose significant computational and memory costs on edge devices. Also, we find that pre-training and meta-training procedures based on source data before deployment improve Meta-CL performance. Finally, to facilitate further research, we provide practical guidelines for researchers and machine learning practitioners implementing Meta-CL on resource-constrained environments and make our benchmark framework and tools publicly available, enabling fair evaluation across both accuracy and system-level metrics. 

**Abstract (ZH)**: 元持续学习（Meta-Continual Learning, Meta-CL）已 emerge 作为通过使用有限标注样本实现持续学习（Continual Learning, CL）以减轻手动标注努力和系统资源需求的颇有前景的方法。然而，尽管现有的方法在图像任务上已显示出成功，它们在来自传感器系统的顺序时间序列数据，特别是音频输入上的有效性仍未被探索。为弥补这一差距，我们使用三种网络架构在五个来自图像和音频模态的数据集中，对六个代表性 Meta-CL 方法进行了全面的基准研究。我们开发了针对边缘设备的端到端 Meta-CL 基准框架 MetaCLBench，用于评估系统开销，并探讨 Meta-CL 方法之间性能、计算成本和内存需求之间的权衡。结果显示，许多 Meta-CL 方法能够同时学习图像和音频模态的新类，但在边缘设备上却会产生明显的计算和内存成本。我们还发现，在部署前基于源数据进行预训练和元训练可以改善 Meta-CL 性能。最后，为了促进进一步研究，我们提供了在资源受限环境中实现 Meta-CL 的实用指南，并将基准框架和工具公开发布，支持基于准确性和系统级指标之间的公平评估。 

---
# Backdoor Detection through Replicated Execution of Outsourced Training 

**Title (ZH)**: 外包训练副本执行的后门检测 

**Authors**: Hengrui Jia, Sierra Wyllie, Akram Bin Sediq, Ahmed Ibrahim, Nicolas Papernot  

**Link**: [PDF](https://arxiv.org/pdf/2504.00170)  

**Abstract**: It is common practice to outsource the training of machine learning models to cloud providers. Clients who do so gain from the cloud's economies of scale, but implicitly assume trust: the server should not deviate from the client's training procedure. A malicious server may, for instance, seek to insert backdoors in the model. Detecting a backdoored model without prior knowledge of both the backdoor attack and its accompanying trigger remains a challenging problem. In this paper, we show that a client with access to multiple cloud providers can replicate a subset of training steps across multiple servers to detect deviation from the training procedure in a similar manner to differential testing. Assuming some cloud-provided servers are benign, we identify malicious servers by the substantial difference between model updates required for backdooring and those resulting from clean training. Perhaps the strongest advantage of our approach is its suitability to clients that have limited-to-no local compute capability to perform training; we leverage the existence of multiple cloud providers to identify malicious updates without expensive human labeling or heavy computation. We demonstrate the capabilities of our approach on an outsourced supervised learning task where $50\%$ of the cloud providers insert their own backdoor; our approach is able to correctly identify $99.6\%$ of them. In essence, our approach is successful because it replaces the signature-based paradigm taken by existing approaches with an anomaly-based detection paradigm. Furthermore, our approach is robust to several attacks from adaptive adversaries utilizing knowledge of our detection scheme. 

**Abstract (ZH)**: 将机器学习模型的训练外包给云提供商是常见的做法。客户端通过这种方式利用了云的成本优势，但前提是信任云服务器不会偏离客户端的训练流程。恶意服务器可能会试图在模型中植入后门。检测带有后门的模型仍是一个挑战性问题，尤其是缺乏关于后门攻击及其触发条件先验知识的情况下。本文展示了客户端如何通过访问多个云提供商，复制部分训练步骤并在多个服务器上执行这些步骤，以类似于差异测试的方式检测训练流程的偏差。假设一些云服务器是无害的，我们可以通过模型更新之间的显著差异来识别恶意服务器，这些差异反映了植入后门所需的更新与干净训练产生的更新不同。我们的方法的一大优势是对训练计算能力有限的客户端而言更为适合；我们利用多个云提供商的存在，无需昂贵的人工标注或大量计算即可识别恶意更新。通过在50%的云提供商中植入后门的一个外包监督学习任务中展示了我们的方法的能力，我们的方法能够正确识别其中99.6%的后门。简而言之，我们的方法成功之处在于将其与现有依赖签名的方法相比，转变为基于异常检测的方法。此外，我们的方法对善于利用我们检测方案知识的适应性攻击具有鲁棒性。 

---
# Does "Reasoning" with Large Language Models Improve Recognizing, Generating, and Reframing Unhelpful Thoughts? 

**Title (ZH)**: 大型语言模型进行“推理”能否改善识别、生成和重新框定无帮助思维？ 

**Authors**: Yilin Qi, Dong Won Lee, Cynthia Breazeal, Hae Won Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.00163)  

**Abstract**: Cognitive Reframing, a core element of Cognitive Behavioral Therapy (CBT), helps individuals reinterpret negative experiences by finding positive meaning. Recent advances in Large Language Models (LLMs) have demonstrated improved performance through reasoning-based strategies. This inspires a promising direction of leveraging the reasoning capabilities of LLMs to improve CBT and mental reframing by simulating the process of critical thinking, potentially enabling more effective recognition, generation, and reframing of cognitive distortions. In this work, we investigate the role of various reasoning methods, including pre-trained reasoning LLMs and augmented reasoning strategies such as CoT and self-consistency in enhancing LLMs' ability to perform cognitive reframing tasks. We find that augmented reasoning methods, even when applied to "outdated" LLMs like GPT-3.5, consistently outperform state-of-the-art pretrained reasoning models on recognizing, generating and reframing unhelpful thoughts. 

**Abstract (ZH)**: 基于推理的认知重框理论：利用大型语言模型增强认知行为疗法中的认知重框技术 

---
# Towards Precise Action Spotting: Addressing Temporal Misalignment in Labels with Dynamic Label Assignment 

**Title (ZH)**: 基于动态标签分配的精确动作检测：解决标签时间错位问题 

**Authors**: Masato Tamura  

**Link**: [PDF](https://arxiv.org/pdf/2504.00149)  

**Abstract**: Precise action spotting has attracted considerable attention due to its promising applications. While existing methods achieve substantial performance by employing well-designed model architecture, they overlook a significant challenge: the temporal misalignment inherent in ground-truth labels. This misalignment arises when frames labeled as containing events do not align accurately with the actual event times, often as a result of human annotation errors or the inherent difficulties in precisely identifying event boundaries across neighboring frames. To tackle this issue, we propose a novel dynamic label assignment strategy that allows predictions to have temporal offsets from ground-truth action times during training, ensuring consistent event spotting. Our method extends the concept of minimum-cost matching, which is utilized in the spatial domain for object detection, to the temporal domain. By calculating matching costs based on predicted action class scores and temporal offsets, our method dynamically assigns labels to the most likely predictions, even when the predicted times of these predictions deviate from ground-truth times, alleviating the negative effects of temporal misalignment in labels. We conduct extensive experiments and demonstrate that our method achieves state-of-the-art performance, particularly in conditions where events are visually distinct and temporal misalignment in labels is common. 

**Abstract (ZH)**: 精确动作检测因其实用前景而受到广泛关注。尽管现有方法通过采用精心设计的模型架构实现了显著性能提升，但它们忽视了一个重要挑战：标注中的时间错位。这种错位源于标注的帧中事件与实际事件时间不准确对齐的问题，这通常是由于人工标注错误或在邻近帧中精确识别事件边界固有的困难所致。为了解决这一问题，我们提出了一种新颖的动力学标签分配策略，该策略允许在训练过程中预测时间与真实动作时间存在一定的时间偏差，从而确保动作的一致识别。我们的方法将空间域中用于对象检测的最小成本匹配概念扩展到了时间域。通过基于预测动作类别得分和时间偏差计算匹配成本，该方法能够动态地将标签分配给最有可能的预测，即使这些预测的时间与真实时间有所偏差，也能减轻标签时间错位带来的负面影响。我们进行了大量实验，并证明了我们的方法在事件视觉上明显不同且标签时间错位普遍的条件下达到了最先进的性能。 

---
# Lorentzian Graph Isomorphic Network 

**Title (ZH)**: 洛伦兹图形同构网络 

**Authors**: Srinitish Srinivasan, Omkumar CU  

**Link**: [PDF](https://arxiv.org/pdf/2504.00142)  

**Abstract**: We introduce the Lorentzian Graph Isomorphic Network (LGIN), a novel graph neural network (GNN) designed to operate in hyperbolic spaces, leveraging the Lorentzian model to enhance graph representation learning. Existing GNNs primarily operate in Euclidean spaces, which can limit their ability to capture hierarchical and multi-relational structures inherent to complex graphs. LGIN addresses this by incorporating curvature-aware aggregation functions that preserve the Lorentzian metric tensor, ensuring embeddings remain constrained within the hyperbolic space by proposing a new update rule that effectively captures both local neighborhood interactions and global structural properties, enabling LGIN to distinguish non-isomorphic graphs with expressiveness at least as powerful as the Weisfeiler-Lehman test. Through extensive evaluation across nine benchmark datasets, including molecular and protein structures, LGIN consistently outperforms or matches state-of-the-art GNNs, demonstrating its robustness and efficacy in modeling complex graph structures. To the best of our knowledge, this is the first study to extend the concept of a powerful graph neural network to Riemannian manifolds, paving the way for future advancements in hyperbolic graph learning. The code for our paper can be found at this https URL. 

**Abstract (ZH)**: 洛伦兹图同构网络：一种基于洛伦兹模型的新型 hyperbolic 图神经网络 

---
# Data-driven Power Loss Identification through Physics-Based Thermal Model Backpropagation 

**Title (ZH)**: 基于物理热模型反向传播的数据驱动功率损失识别 

**Authors**: Mattia Scarpa, Francesco Pase, Ruggero Carli, Mattia Bruschetta, Franscesco Toso  

**Link**: [PDF](https://arxiv.org/pdf/2504.00133)  

**Abstract**: Digital twins for power electronics require accurate power losses whose direct measurements are often impractical or impossible in real-world applications. This paper presents a novel hybrid framework that combines physics-based thermal modeling with data-driven techniques to identify and correct power losses accurately using only temperature measurements. Our approach leverages a cascaded architecture where a neural network learns to correct the outputs of a nominal power loss model by backpropagating through a reduced-order thermal model. We explore two neural architectures, a bootstrapped feedforward network, and a recurrent neural network, demonstrating that the bootstrapped feedforward approach achieves superior performance while maintaining computational efficiency for real-time applications. Between the interconnection, we included normalization strategies and physics-guided training loss functions to preserve stability and ensure physical consistency. Experimental results show that our hybrid model reduces both temperature estimation errors (from 7.2+-6.8°C to 0.3+-0.3°C) and power loss prediction errors (from 5.4+-6.6W to 0.2+-0.3W) compared to traditional physics-based approaches, even in the presence of thermal model uncertainties. This methodology allows us to accurately estimate power losses without direct measurements, making it particularly helpful for real-time industrial applications where sensor placement is hindered by cost and physical limitations. 

**Abstract (ZH)**: 数字孪生技术在电力电子中的应用要求准确的功率损耗数据，而在实际应用中直接测量这些数据往往不现实或不可能。本文提出了一种新的混合框架，该框架结合了基于物理的热学建模与数据驱动技术，仅通过温度测量即可准确识别和校正功率损耗。该方法采用嵌套架构，其中神经网络通过反向传播通过降阶热模型来学习校正名义功率损耗模型的输出。我们探索了两种神经网络架构——自助前向网络和递归神经网络，结果显示自助前向网络在保持实时应用所需计算效率的同时，实现了更优的性能。在两者之间，我们还包含了归一化策略和基于物理的训练损失函数，以保持模型的稳定性和物理一致性。实验结果显示，与传统的基于物理的方法相比，我们的混合模型在存在热模型不确定性的情况下，能够降低温度估计误差（从7.2±6.8°C降至0.3±0.3°C）和功率损耗预测误差（从5.4±6.6W降至0.2±0.3W），从而特别适用于由于成本和物理限制而受到传感器放置限制的实时工业应用。 

---
# Times2D: Multi-Period Decomposition and Derivative Mapping for General Time Series Forecasting 

**Title (ZH)**: Times2D: 多时期分解和导数映射通用时间序列预测 

**Authors**: Reza Nematirad, Anil Pahwa, Balasubramaniam Natarajan  

**Link**: [PDF](https://arxiv.org/pdf/2504.00118)  

**Abstract**: Time series forecasting is an important application in various domains such as energy management, traffic planning, financial markets, meteorology, and medicine. However, real-time series data often present intricate temporal variability and sharp fluctuations, which pose significant challenges for time series forecasting. Previous models that rely on 1D time series representations usually struggle with complex temporal variations. To address the limitations of 1D time series, this study introduces the Times2D method that transforms the 1D time series into 2D space. Times2D consists of three main parts: first, a Periodic Decomposition Block (PDB) that captures temporal variations within a period and between the same periods by converting the time series into a 2D tensor in the frequency domain. Second, the First and Second Derivative Heatmaps (FSDH) capture sharp changes and turning points, respectively. Finally, an Aggregation Forecasting Block (AFB) integrates the output tensors from PDB and FSDH for accurate forecasting. This 2D transformation enables the utilization of 2D convolutional operations to effectively capture long and short characteristics of the time series. Comprehensive experimental results across large-scale data in the literature demonstrate that the proposed Times2D model achieves state-of-the-art performance in both short-term and long-term forecasting. The code is available in this repository: this https URL. 

**Abstract (ZH)**: 时间序列 forecasting 是能源管理、交通规划、金融市场、气象学和医学等领域中的一项重要应用。然而，真实世界的时间序列数据 often 呈现出复杂的时域变异性以及尖锐的波动，这对时间序列 forecasting 带来了巨大挑战。依赖于 1D 时间序列表示的先前模型通常难以处理复杂的时域变化。为了应对 1D 时间序列的局限性，本文提出了 Times2D 方法，将 1D 时间序列转换为 2D 空间。Times2D 由三个主要部分组成：首先，周期分解块 (PDB) 将时间序列在频域中转换为 2D 张量，以捕捉同一周期内和不同周期之间的时域变化。其次，一阶和二阶导数热图 (FSDH) 分别捕捉尖锐变化和转折点。最后，聚合预测块 (AFB) 将 PDB 和 FSDH 的输出张量进行集成，以实现准确的预测。这种 2D 转换使利用 2D 卷积操作能够有效捕捉时间序列的长期和短期特征。在文献中的大规模数据集上进行的综合实验证明，所提出的 Times2D 模型在短、长期 forecasting 领域均达到了最先进的性能。代码见本仓库：this https URL。 

---
# Assessing Code Understanding in LLMs 

**Title (ZH)**: 评估LLM中的代码理解能力 

**Authors**: Cosimo Laneve, Alvise Spanò, Dalila Ressi, Sabina Rossi, Michele Bugliesi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00065)  

**Abstract**: We present an empirical evaluation of Large Language Models in code understanding associated with non-trivial, semantic-preserving program transformations such as copy propagation or constant folding. Our findings show that LLMs fail to judge semantic equivalence in approximately 41\% of cases when no context is provided and in 29\% when given a simple generic context. To improve accuracy, we advocate integrating LLMs with code-optimization tools to enhance training and facilitate more robust program understanding. 

**Abstract (ZH)**: 我们对大型语言模型在代码理解方面的实证评估，特别是在处理诸如复制传播或常数折叠等非平凡、语义保持的程序变换方面的能力。研究发现，在没有提供上下文的情况下，大型语言模型在约41%的案例中未能判断语义等价性；即使提供了简单的通用上下文，这一比例也达到了29%。为了提高准确性，我们建议将大型语言模型与代码优化工具集成，以增强训练并促进更稳健的程序理解。 

---
# Evaluating the Feasibility and Accuracy of Large Language Models for Medical History-Taking in Obstetrics and Gynecology 

**Title (ZH)**: 评估大型语言模型在妇产科学科医疗病史采集中的可行性与准确性 

**Authors**: Dou Liu, Ying Long, Sophia Zuoqiu, Tian Tang, Rong Yin  

**Link**: [PDF](https://arxiv.org/pdf/2504.00061)  

**Abstract**: Effective physician-patient communications in pre-diagnostic environments, and most specifically in complex and sensitive medical areas such as infertility, are critical but consume a lot of time and, therefore, cause clinic workflows to become inefficient. Recent advancements in Large Language Models (LLMs) offer a potential solution for automating conversational medical history-taking and improving diagnostic accuracy. This study evaluates the feasibility and performance of LLMs in those tasks for infertility cases. An AI-driven conversational system was developed to simulate physician-patient interactions with ChatGPT-4o and ChatGPT-4o-mini. A total of 70 real-world infertility cases were processed, generating 420 diagnostic histories. Model performance was assessed using F1 score, Differential Diagnosis (DDs) Accuracy, and Accuracy of Infertility Type Judgment (ITJ). ChatGPT-4o-mini outperformed ChatGPT-4o in information extraction accuracy (F1 score: 0.9258 vs. 0.9029, p = 0.045, d = 0.244) and demonstrated higher completeness in medical history-taking (97.58% vs. 77.11%), suggesting that ChatGPT-4o-mini is more effective in extracting detailed patient information, which is critical for improving diagnostic accuracy. In contrast, ChatGPT-4o performed slightly better in differential diagnosis accuracy (2.0524 vs. 2.0048, p > 0.05). ITJ accuracy was higher in ChatGPT-4o-mini (0.6476 vs. 0.5905) but with lower consistency (Cronbach's $\alpha$ = 0.562), suggesting variability in classification reliability. Both models demonstrated strong feasibility in automating infertility history-taking, with ChatGPT-4o-mini excelling in completeness and extraction accuracy. In future studies, expert validation for accuracy and dependability in a clinical setting, AI model fine-tuning, and larger datasets with a mix of cases of infertility have to be prioritized. 

**Abstract (ZH)**: 有效诊断前的医患沟通在预诊环境中至关重要，尤其是在不孕不育等复杂和敏感的医疗领域，但这些沟通耗费大量时间，导致 clinics 工作流程效率低下。近年来大语言模型（LLMs）的进展为自动化问诊和提高诊断准确性提供了潜在解决方案。本研究评估了大语言模型在不孕不育病例中进行任务的可行性和性能。开发了一个以ChatGPT-4o和ChatGPT-4o-mini为基础的AI驱动对话系统，处理了70例实际的不孕不育病例，生成了420份诊断病史。模型性能通过F1分数、鉴别诊断准确性和不孕不育类型判断准确率（ITJ）进行评估。ChatGPT-4o-mini在信息提取准确性（F1分数：0.9258 vs. 0.9029，p=0.045，d=0.244）和医学病史提取完整性（97.58% vs. 77.11%）方面表现更优，表明ChatGPT-4o-mini在提取详细患者信息方面更有效，这对于提高诊断准确性至关重要。相比之下，ChatGPT-4o在鉴别诊断准确率方面略优（2.0524 vs. 2.0048，p>0.05）。ChatGPT-4o-mini在ITJ准确率方面更高（0.6476 vs. 0.5905），但一致性较低（Cronbach's α = 0.562），表明分类可靠性存在变异性。两种模型在自动化不孕不育病史提取方面具有较强的可行性，ChatGPT-4o-mini在完整性与提取准确性方面表现更优。未来研究需优先考虑临床环境下的专家验证、AI模型微调以及包含不同不孕不育病例的更大数据集。 

---
# CF-CAM: Gradient Perturbation Mitigation and Feature Stabilization for Reliable Interpretability 

**Title (ZH)**: CF-CAM: 梯度扰动缓解与特征稳定化以实现可靠的可解释性 

**Authors**: Hongjie He, Xu Pan, Yudong Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.00060)  

**Abstract**: As deep learning continues to advance, the opacity of neural network decision-making remains a critical challenge, limiting trust and applicability in high-stakes domains. Class Activation Mapping (CAM) techniques have emerged as a key approach to visualizing model decisions, yet existing methods face inherent trade-offs. Gradient-based CAM variants suffer from sensitivity to gradient perturbations, leading to unstable and unreliable explanations. Conversely, gradient-free approaches mitigate gradient instability but incur significant computational overhead and inference latency. To address these limitations, we propose Cluster Filter Class Activation Map (CF-CAM), a novel framework that reintroduces gradient-based weighting while enhancing robustness against gradient noise. CF-CAM employs a hierarchical importance weighting strategy to balance discriminative feature preservation and noise elimination. A density-aware channel clustering via Density-Based Spatial Clustering of Applications with Noise (DBSCAN) groups semantically relevant feature channels and discard noise-prone activations. Additionally, cluster-conditioned gradient filtering leverages bilateral filters to refine gradient signals, preserving edge-aware localization while suppressing noise impact. Experiment results demonstrate that CF-CAM achieves superior interpretability performance while maintaining resilience to gradient perturbations, outperforming state-of-the-art CAM methods in faithfulness and robustness. By effectively mitigating gradient instability without excessive computational cost, CF-CAM provides a reliable solution for enhancing the interpretability of deep neural networks in critical applications such as medical diagnosis and autonomous driving. 

**Abstract (ZH)**: 基于聚类滤波的类激活图（Cluster Filter Class Activation Map, CF-CAM）：增强深层神经网络的可解释性与鲁棒性 

---
# GAL-MAD: Towards Explainable Anomaly Detection in Microservice Applications Using Graph Attention Networks 

**Title (ZH)**: GAL-MAD：基于图注意力网络的可解释微服务应用异常检测 

**Authors**: Lahiru Akmeemana, Chamodya Attanayake, Husni Faiz, Sandareka Wickramanayake  

**Link**: [PDF](https://arxiv.org/pdf/2504.00058)  

**Abstract**: The transition to microservices has revolutionized software architectures, offering enhanced scalability and modularity. However, the distributed and dynamic nature of microservices introduces complexities in ensuring system reliability, making anomaly detection crucial for maintaining performance and functionality. Anomalies stemming from network and performance issues must be swiftly identified and addressed. Existing anomaly detection techniques often rely on statistical models or machine learning methods that struggle with the high-dimensional, interdependent data inherent in microservice applications. Current techniques and available datasets predominantly focus on system traces and logs, limiting their ability to support advanced detection models. This paper addresses these gaps by introducing the RS-Anomic dataset generated using the open-source RobotShop microservice application. The dataset captures multivariate performance metrics and response times under normal and anomalous conditions, encompassing ten types of anomalies. We propose a novel anomaly detection model called Graph Attention and LSTM-based Microservice Anomaly Detection (GAL-MAD), leveraging Graph Attention and Long Short-Term Memory architectures to capture spatial and temporal dependencies in microservices. We utilize SHAP values to localize anomalous services and identify root causes to enhance explainability. Experimental results demonstrate that GAL-MAD outperforms state-of-the-art models on the RS-Anomic dataset, achieving higher accuracy and recall across varying anomaly rates. The explanations provide actionable insights into service anomalies, which benefits system administrators. 

**Abstract (ZH)**: 微服务转型重塑了软件架构，提升了可扩展性和模块性。然而，微服务的分布式和动态特性带来了确保系统可靠性的复杂性，这使得异常检测对于保持性能和功能至关重要。源自网络和性能问题的异常必须迅速被识别和处理。现有异常检测技术通常依赖于统计模型或机器学习方法，这些方法在处理微服务应用中存在的高维和相互依赖的数据时表现不佳。当前的技术和可用的数据集主要集中在系统跟踪和日志上，限制了它们支持高级检测模型的能力。本文通过引入基于开源RobotShop微服务应用生成的RS-Anomic数据集来填补这些空白。该数据集在正常和异常条件下捕获了多变量性能指标和响应时间，并包括了十种类型的异常。我们提出了一种新的异常检测模型——基于图注意力和长短期记忆的微服务异常检测（GAL-MAD），利用图注意力和长短期记忆架构来捕捉微服务中的空间和时间依赖性。我们使用SHAP值来定位异常服务并识别根本原因，以提高解释性。实验结果表明，GAL-MAD在RS-Anomic数据集上优于最先进的模型，即使在不同异常率的情况下也能实现更高的准确性和召回率。这些解释为系统管理员提供了可操作的洞察，帮助他们更好地理解和处理服务异常。 

---
# Integrating Large Language Models with Human Expertise for Disease Detection in Electronic Health Records 

**Title (ZH)**: 将大型语言模型与人类专长集成以检测电子健康记录中的疾病 

**Authors**: Jie Pan, Seungwon Lee, Cheligeer Cheligeer, Elliot A. Martin, Kiarash Riazi, Hude Quan, Na Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.00053)  

**Abstract**: Objective: Electronic health records (EHR) are widely available to complement administrative data-based disease surveillance and healthcare performance evaluation. Defining conditions from EHR is labour-intensive and requires extensive manual labelling of disease outcomes. This study developed an efficient strategy based on advanced large language models to identify multiple conditions from EHR clinical notes. Methods: We linked a cardiac registry cohort in 2015 with an EHR system in Alberta, Canada. We developed a pipeline that leveraged a generative large language model (LLM) to analyze, understand, and interpret EHR notes by prompts based on specific diagnosis, treatment management, and clinical guidelines. The pipeline was applied to detect acute myocardial infarction (AMI), diabetes, and hypertension. The performance was compared against clinician-validated diagnoses as the reference standard and widely adopted International Classification of Diseases (ICD) codes-based methods. Results: The study cohort accounted for 3,088 patients and 551,095 clinical notes. The prevalence was 55.4%, 27.7%, 65.9% and for AMI, diabetes, and hypertension, respectively. The performance of the LLM-based pipeline for detecting conditions varied: AMI had 88% sensitivity, 63% specificity, and 77% positive predictive value (PPV); diabetes had 91% sensitivity, 86% specificity, and 71% PPV; and hypertension had 94% sensitivity, 32% specificity, and 72% PPV. Compared with ICD codes, the LLM-based method demonstrated improved sensitivity and negative predictive value across all conditions. The monthly percentage trends from the detected cases by LLM and reference standard showed consistent patterns. 

**Abstract (ZH)**: 目标：电子健康记录（EHR）广泛可用，可以补充基于行政数据的疾病监测和医疗服务绩效评价。从EHR中定义条件劳动强度大，需要进行广泛的疾病结果手工标注。本研究基于先进的大语言模型开发了一种高效策略，用于识别EHR临床笔记中的多种条件。方法：我们将2015年的冠心病注册队列与加拿大的艾伯塔省EHR系统连接。我们开发了一个流水线，利用生成性大语言模型（LLM）通过基于特定诊断、治疗管理及临床指南的提示来分析、理解和解释EHR笔记。该流水线应用于检测急性心肌梗死（AMI）、糖尿病和高血压。性能与临床验证诊断和广泛应用的国际疾病分类（ICD）编码方法进行了比较。结果：研究队列包括3088名患者和551095份临床笔记。AMIs、糖尿病和高血压的患病率分别为55.4%、27.7%和65.9%。基于LLM的流水线在检测条件方面的性能不同：AMI的敏感性为88%，特异性为63%，阳性预测值为77%；糖尿病的敏感性为91%，特异性为86%，阳性预测值为71%；高血压的敏感性为94%，特异性为32%，阳性预测值为72%。与ICD编码相比，基于LLM的方法在所有条件下均显示提高了敏感性和阴性预测值。通过LLM和参考标准检测的确诊病例的月百分比趋势显示出一致的模式。 

---
# The Cursive Transformer 

**Title (ZH)**: 连笔变压器 

**Authors**: Sam Greydanus, Zachary Wimpee  

**Link**: [PDF](https://arxiv.org/pdf/2504.00051)  

**Abstract**: Transformers trained on tokenized text, audio, and images can generate high-quality autoregressive samples. But handwriting data, represented as sequences of pen coordinates, remains underexplored. We introduce a novel tokenization scheme that converts pen stroke offsets to polar coordinates, discretizes them into bins, and then turns them into sequences of tokens with which to train a standard GPT model. This allows us to capture complex stroke distributions without using any specialized architectures (eg. the mixture density network or the self-advancing ASCII attention head from Graves 2014). With just 3,500 handwritten words and a few simple data augmentations, we are able to train a model that can generate realistic cursive handwriting. Our approach is simpler and more performant than previous RNN-based methods. 

**Abstract (ZH)**: 基于标记化文本、音频和图像训练的变压器可以生成高质量的自回归样本。但作为笔画坐标序列的手写数据仍处于未充分探索状态。我们提出了一种新颖的标记化方案，将笔画偏移转换为极坐标，对其离散化分箱，然后使用这些标记序列来训练标准的GPT模型。这使我们能够在不使用任何专有架构（例如混合密度网络或格瑞夫斯2014年的自推进ASCII注意力头）的情况下捕捉复杂的笔画分布。仅使用3,500个手写词和少量简单的数据增强，我们就能训练一个能够生成逼真手写体模型。我们的方法在RNN基方法上更加简洁且性能更优。 

---
# JudgeLRM: Large Reasoning Models as a Judge 

**Title (ZH)**: JudgeLRM：大型推理模型作为法官 

**Authors**: Nuo Chen, Zhiyuan Hu, Qingyun Zou, Jiaying Wu, Qian Wang, Bryan Hooi, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2504.00050)  

**Abstract**: The rise of Large Language Models (LLMs) as evaluators offers a scalable alternative to human annotation, yet existing Supervised Fine-Tuning (SFT) for judges approaches often fall short in domains requiring complex reasoning. In this work, we investigate whether LLM judges truly benefit from enhanced reasoning capabilities. Through a detailed analysis of reasoning requirements across evaluation tasks, we reveal a negative correlation between SFT performance gains and the proportion of reasoning-demanding samples - highlighting the limitations of SFT in such scenarios. To address this, we introduce JudgeLRM, a family of judgment-oriented LLMs trained using reinforcement learning (RL) with judge-wise, outcome-driven rewards. JudgeLRM models consistently outperform both SFT-tuned and state-of-the-art reasoning models. Notably, JudgeLRM-3B surpasses GPT-4, and JudgeLRM-7B outperforms DeepSeek-R1 by 2.79% in F1 score, particularly excelling in judge tasks requiring deep reasoning. 

**Abstract (ZH)**: 大规模语言模型作为评估者崛起提供了规模化的人类标注替代方案，但在需要复杂推理的领域，现有的监督微调（SFT）方法往往表现不佳。本研究探索大规模语言模型评估者是否真的可以从增强的推理能力中受益。通过对评估任务中推理需求的详细分析，我们揭示了SFT性能改进与推理需求样本比例之间的负相关性，突显了SFT在这些场景中的局限性。为了解决这一问题，我们提出了一种使用基于奖励的强化学习（RL）训练的JudgeLRM系列判断导向的大规模语言模型。JudgeLRM模型在判断任务中的一致性评估中表现出色，优于SFT调优的大规模语言模型和最先进的推理模型。值得注意的是，JudgeLRM-3B超过了GPT-4，而JudgeLRM-7B在F1分数上比DeepSeek-R1高出了2.79%，特别是在需要深度推理的判断任务中表现尤为出色。 

---
# Distill-C: Enhanced NL2SQL via Distilled Customization with LLMs 

**Title (ZH)**: Distill-C: 通过LLM辅助精简定制增强NL2SQL 

**Authors**: Cong Duy Vu Hoang, Gioacchino Tangari, Clemence Lanfranchi, Dalu Guo, Paul Cayet, Steve Siu, Don Dharmasiri, Yuan-Fang Li, Long Duong, Damien Hilloulin, Rhicheek Patra, Sungpack Hong, Hassan Chafi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00048)  

**Abstract**: The growing adoption of large language models (LLMs) in business applications has amplified interest in Natural Language to SQL (NL2SQL) solutions, in which there is competing demand for high performance and efficiency. Domain- and customer-specific requirements further complicate the problem. To address this conundrum, we introduce Distill-C, a distilled customization framework tailored for NL2SQL tasks. Distill-C utilizes large teacher LLMs to produce high-quality synthetic data through a robust and scalable pipeline. Finetuning smaller and open-source LLMs on this synthesized data enables them to rival or outperform teacher models an order of magnitude larger. Evaluated on multiple challenging benchmarks, Distill-C achieves an average improvement of 36% in execution accuracy compared to the base models from three distinct LLM families. Additionally, on three internal customer benchmarks, Distill-C demonstrates a 22.6% performance improvement over the base models. Our results demonstrate that Distill-C is an effective, high-performing and generalizable approach for deploying lightweight yet powerful NL2SQL models, delivering exceptional accuracies while maintaining low computational cost. 

**Abstract (ZH)**: 大型语言模型在企业应用中的不断 adoption 已加剧了对自然语言转结构查询语言（NL2SQL）解决方案的需求，这类解决方案在高性能和效率之间存在竞争需求。特定领域和客户的要求进一步复杂化了这一问题。为应对这一挑战，我们引入了 Distill-C，这是一种针对 NL2SQL 任务定制的精简框架。Distill-C 利用大型教师语言模型通过一个稳健且可扩展的流水线生成高质量的合成数据。在这些合成数据上对较小的开源语言模型进行微调，使它们能够与或超越比它们大一个数量级的教师模型。在多个具有挑战性的基准测试中，Distill-C 的执行准确率平均提高了 36%，相对于三个不同语言模型家族的基础模型。此外，在三个内部客户基准测试中，Distill-C 的性能比基础模型提高了 22.6%。我们的结果表明，Distill-C 是一种有效的、高性能且可泛化的轻量级但强大的 NL2SQL 模型部署方法，能够在保持较低计算成本的同时实现出色的效果。 

---
# EAP4EMSIG -- Enhancing Event-Driven Microscopy for Microfluidic Single-Cell Analysis 

**Title (ZH)**: EAP4EMSIG -- 增强事件驱动显微镜技术用于微流控单细胞分析 

**Authors**: Nils Friederich, Angelo Jovin Yamachui Sitcheu, Annika Nassal, Erenus Yildiz, Matthias Pesch, Maximilian Beichter, Lukas Scholtes, Bahar Akbaba, Thomas Lautenschlager, Oliver Neumann, Dietrich Kohlheyer, Hanno Scharr, Johannes Seiffarth, Katharina Nöh, Ralf Mikut  

**Link**: [PDF](https://arxiv.org/pdf/2504.00047)  

**Abstract**: Microfluidic Live-Cell Imaging yields data on microbial cell factories. However, continuous acquisition is challenging as high-throughput experiments often lack realtime insights, delaying responses to stochastic events. We introduce three components in the Experiment Automation Pipeline for Event-Driven Microscopy to Smart Microfluidic Single-Cell Analysis: a fast, accurate Deep Learning autofocusing method predicting the focus offset, an evaluation of real-time segmentation methods and a realtime data analysis dashboard. Our autofocusing achieves a Mean Absolute Error of 0.0226\textmu m with inference times below 50~ms. Among eleven Deep Learning segmentation methods, Cellpose~3 reached a Panoptic Quality of 93.58\%, while a distance-based method is fastest (121~ms, Panoptic Quality 93.02\%). All six Deep Learning Foundation Models were unsuitable for real-time segmentation. 

**Abstract (ZH)**: 微流控活细胞成像提供了微生物细胞工厂的数据。然而，连续获取数据颇具挑战，因为高通量实验往往缺乏实时洞察，延误了对随机事件的响应。我们提出了事件驱动显微镜到智能微流控单细胞分析实验自动化管道的三个组成部分：一种快速准确的深度学习自动对焦方法，用于预测焦深偏移，实时分割方法的评估，以及实时数据分析仪表盘。我们的自动对焦方法的绝对平均误差为0.0226μm，推理时间低于50毫秒。在 eleven 种深度学习分割方法中，Cellpose 达到了 93.58% 的全景质量，而基于距离的方法最快（121 毫秒，全景质量 93.02%）。所有六种深度学习基础模型都不适合实时分割。 

---
# Multi-Stakeholder Disaster Insights from Social Media Using Large Language Models 

**Title (ZH)**: 使用大型语言模型从社交媒体获取多方灾害见解 

**Authors**: Loris Belcastro, Cristian Cosentino, Fabrizio Marozzo, Merve Gündüz-Cüre, Şule Öztürk-Birim  

**Link**: [PDF](https://arxiv.org/pdf/2504.00046)  

**Abstract**: In recent years, social media has emerged as a primary channel for users to promptly share feedback and issues during disasters and emergencies, playing a key role in crisis management. While significant progress has been made in collecting and analyzing social media content, there remains a pressing need to enhance the automation, aggregation, and customization of this data to deliver actionable insights tailored to diverse stakeholders, including the press, police, EMS, and firefighters. This effort is essential for improving the coordination of activities such as relief efforts, resource distribution, and media communication. This paper presents a methodology that leverages the capabilities of LLMs to enhance disaster response and management. Our approach combines classification techniques with generative AI to bridge the gap between raw user feedback and stakeholder-specific reports. Social media posts shared during catastrophic events are analyzed with a focus on user-reported issues, service interruptions, and encountered challenges. We employ full-spectrum LLMs, using analytical models like BERT for precise, multi-dimensional classification of content type, sentiment, emotion, geolocation, and topic. Generative models such as ChatGPT are then used to produce human-readable, informative reports tailored to distinct audiences, synthesizing insights derived from detailed classifications. We compare standard approaches, which analyze posts directly using prompts in ChatGPT, to our advanced method, which incorporates multi-dimensional classification, sub-event selection, and tailored report generation. Our methodology demonstrates superior performance in both quantitative metrics, such as text coherence scores and latent representations, and qualitative assessments by automated tools and field experts, delivering precise insights for diverse disaster response stakeholders. 

**Abstract (ZH)**: 近年来，社交媒体已成为用户在灾难和紧急事件中及时分享反馈和问题的主要渠道，在危机管理中发挥着关键作用。尽管在收集和分析社交媒体内容方面取得了显著进展，但仍迫切需要增强这些数据的自动化、聚合和定制化，以提供针对不同利益相关者的个性化行动建议，包括媒体、警察、急救服务和消防员。这种努力对于提高如救援工作、资源分配和媒体沟通等活动的协调性至关重要。本文提出了一种利用大型语言模型（LLM）能力的方法，以增强灾难响应和管理。我们的方法结合了分类技术和生成AI，以填补原始用户反馈与特定利益相关者报告之间的差距。在灾难事件期间共享的社交媒体帖子被分析，重点关注用户报告的问题、服务中断以及遇到的挑战。我们使用全谱LLMs，采用如BERT的分析模型对内容类型、情感、情绪、地理位置和主题进行精确的多维度分类。然后使用如ChatGPT的生成模型生成针对不同受众的人类可读且信息丰富的报告，综合详细的分类洞察。我们将基于ChatGPT直接分析帖子的标准方法与我们的方法进行了比较，该方法结合了多维度分类、分事件选择和定制报告生成。我们的方法在定量指标（如文本连贯性评分和潜在表示）和自动化工具及现场专家的定性评估中都展现了更优的表现，为不同的灾难响应相关方提供精确的洞察。 

---
# CrossWordBench: Evaluating the Reasoning Capabilities of LLMs and LVLMs with Controllable Puzzle Generation 

**Title (ZH)**: CrossWordBench：通过可控谜题生成评估大语言模型和大型向量语言模型的推理能力 

**Authors**: Jixuan Leng, Chengsong Huang, Langlin Huang, Bill Yuchen Lin, William W. Cohen, Haohan Wang, Jiaxin Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00043)  

**Abstract**: Existing reasoning evaluation frameworks for Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) predominantly either assess text-based reasoning or vision-language understanding capabilities, with limited dynamic interplay between textual and visual constraints. To address this limitation, we introduce CrossWordBench, a benchmark designed to evaluate the reasoning capabilities of both LLMs and LVLMs through the medium of crossword puzzles-a task requiring multimodal adherence to semantic constraints from text-based clues and intersectional constraints from visual grid structures. CrossWordBench leverages a controllable puzzle generation framework that produces puzzles in multiple formats (text and image) and offers different evaluation strategies ranging from direct puzzle solving to interactive modes. Our extensive evaluation of over 20 models reveals that reasoning LLMs outperform non-reasoning models substantially by effectively leveraging crossing-letter constraints. We further demonstrate that LVLMs struggle with the task, showing a strong correlation between their puzzle-solving performance and grid-parsing accuracy. Our findings offer insights into the limitations of the reasoning capabilities of current LLMs and LVLMs, and provide an effective approach for creating multimodal constrained tasks for future evaluations. 

**Abstract (ZH)**: 现有的大规模语言模型(LLMs)和大规模视觉-语言模型(LVLMs)推理评估框架主要评估文本推理或视觉-语言理解能力，但文本和视觉约束之间的动态交互有限。为解决这一问题，我们引入了CrossWordBench，这是一个通过填字谜题评估LLMs和LVLMs推理能力的基准，填字谜题需要同时遵循文本线索的语义约束和图形结构的交叉约束。CrossWordBench 利用了一个可控的谜题生成框架，生成多种格式（文本和图像）的谜题，并提供从直接解谜到互动模式的不同评估策略。我们的广泛评估表明，具备推理能力的LLMs在利用交叉字母约束方面显著优于不具备推理能力的模型。此外，我们还展示了LVLMs在任务中的困难，并发现它们的解谜性能与图形解析准确性之间存在强烈相关性。我们的研究结果揭示了当前LLMs和LVLMs推理能力的局限性，并为未来评估提供了有效的多模态约束任务设计方法。 

---
# Quantum Methods for Managing Ambiguity in Natural Language Processing 

**Title (ZH)**: 量子方法在自然语言处理中管理歧义的应用 

**Authors**: Jurek Eisinger, Ward Gauderis, Lin de Huybrecht, Geraint A. Wiggins  

**Link**: [PDF](https://arxiv.org/pdf/2504.00040)  

**Abstract**: The Categorical Compositional Distributional (DisCoCat) framework models meaning in natural language using the mathematical framework of quantum theory, expressed as formal diagrams. DisCoCat diagrams can be associated with tensor networks and quantum circuits. DisCoCat diagrams have been connected to density matrices in various contexts in Quantum Natural Language Processing (QNLP). Previous use of density matrices in QNLP entails modelling ambiguous words as probability distributions over more basic words (the word \texttt{queen}, e.g., might mean the reigning queen or the chess piece). In this article, we investigate using probability distributions over processes to account for syntactic ambiguity in sentences. The meanings of these sentences are represented by density matrices. We show how to create probability distributions on quantum circuits that represent the meanings of sentences and explain how this approach generalises tasks from the literature. We conduct an experiment to validate the proposed theory. 

**Abstract (ZH)**: 基于量子理论的分类组合分布（DisCoCat）框架使用形式图表来表示自然语言的意义。DisCoCat图表可以与张量网络和量子电路相联系。在量子自然语言处理（QNLP）的多种背景下，DisCoCat图表与密度矩阵相关联。在先前于QNLP中的密度矩阵使用中，模糊词被建模为更基本词的概率分布（例如，“queen”这个词可能指的是在位的女王或棋盘上的棋子）。在本文中，我们研究使用过程的概率分布来解释句子的句法模糊性。这些句子的意义由密度矩阵表示。我们展示了如何在表示句子意义的量子电路上创建概率分布，并解释了这种方法如何推广文献中的任务。我们进行了一项实验来验证提出理论的有效性。 

---
# Revisiting the Relationship between Adversarial and Clean Training: Why Clean Training Can Make Adversarial Training Better 

**Title (ZH)**: 重新审视对抗训练与干净训练之间的关系：为什么干净训练可以使对抗训练更好 

**Authors**: MingWei Zhou, Xiaobing Pei  

**Link**: [PDF](https://arxiv.org/pdf/2504.00038)  

**Abstract**: Adversarial training (AT) is an effective technique for enhancing adversarial robustness, but it usually comes at the cost of a decline in generalization ability. Recent studies have attempted to use clean training to assist adversarial training, yet there are contradictions among the conclusions. We comprehensively summarize the representative strategies and, with a focus on the multi - view hypothesis, provide a unified explanation for the contradictory phenomena among different studies. In addition, we conduct an in - depth analysis of the knowledge combinations transferred from clean - trained models to adversarially - trained models in previous studies, and find that they can be divided into two categories: reducing the learning difficulty and providing correct guidance. Based on this finding, we propose a new idea of leveraging clean training to further improve the performance of advanced AT this http URL reveal that the problem of generalization degradation faced by AT partly stems from the difficulty of adversarial training in learning certain sample features, and this problem can be alleviated by making full use of clean training. 

**Abstract (ZH)**: 对抗训练（AT）是一种有效的增强对抗鲁棒性的技术，但通常会以牺牲泛化能力为代价。最近的研究试图通过干净的训练来辅助对抗训练，然而不同研究的结论之间存在矛盾。我们全面总结了代表性策略，并以多视角假说为重点，提供了不同研究中矛盾现象的统一解释。此外，我们深入分析了之前研究中干净训练模型向对抗训练模型转移的知识组合，并发现它们可以分为两类：减少学习难度和提供正确指导。基于这一发现，我们提出了一个新的思路，即利用干净训练进一步提高高级AT的性能：对抗训练面临的泛化能力下降问题部分源自于学习某些样本特征的难度，这个问题可以通过充分利用干净训练来缓解。 

---
# ViT-Linearizer: Distilling Quadratic Knowledge into Linear-Time Vision Models 

**Title (ZH)**: ViT-线性化器：将二次知识 distilled 到线性时间视觉模型中 

**Authors**: Guoyizhe Wei, Rama Chellappa  

**Link**: [PDF](https://arxiv.org/pdf/2504.00037)  

**Abstract**: Vision Transformers (ViTs) have delivered remarkable progress through global self-attention, yet their quadratic complexity can become prohibitive for high-resolution inputs. In this work, we present ViT-Linearizer, a cross-architecture distillation framework that transfers rich ViT representations into a linear-time, recurrent-style model. Our approach leverages 1) activation matching, an intermediate constraint that encourages student to align its token-wise dependencies with those produced by the teacher, and 2) masked prediction, a contextual reconstruction objective that requires the student to predict the teacher's representations for unseen (masked) tokens, to effectively distill the quadratic self-attention knowledge into the student while maintaining efficient complexity. Empirically, our method provides notable speedups particularly for high-resolution tasks, significantly addressing the hardware challenges in inference. Additionally, it also elevates Mamba-based architectures' performance on standard vision benchmarks, achieving a competitive 84.3% top-1 accuracy on ImageNet with a base-sized model. Our results underscore the good potential of RNN-based solutions for large-scale visual tasks, bridging the gap between theoretical efficiency and real-world practice. 

**Abstract (ZH)**: Vision Transformers线性化器（ViT-Linearizer）：一种跨架构知识蒸馏框架 

---
# Improving Diseases Predictions Utilizing External Bio-Banks 

**Title (ZH)**: 利用外部生物银行提高疾病预测准确性 

**Authors**: Hido Pinto, Eran Segal  

**Link**: [PDF](https://arxiv.org/pdf/2504.00036)  

**Abstract**: Machine learning has been successfully used in critical domains, such as medicine. However, extracting meaningful insights from biomedical data is often constrained by the lack of their available disease labels. In this research, we demonstrate how machine learning can be leveraged to enhance explainability and uncover biologically meaningful associations, even when predictive improvements in disease modeling are limited. We train LightGBM models from scratch on our dataset (10K) to impute metabolomics features and apply them to the UK Biobank (UKBB) for downstream analysis. The imputed metabolomics features are then used in survival analysis to assess their impact on disease-related risk factors. As a result, our approach successfully identified biologically relevant connections that were not previously known to the predictive models. Additionally, we applied a genome-wide association study (GWAS) on key metabolomics features, revealing a link between vascular dementia and smoking. Although being a well-established epidemiological relationship, this link was not embedded in the model's training data, which validated the method's ability to extract meaningful signals. Furthermore, by integrating survival models as inputs in the 10K data, we uncovered associations between metabolic substances and obesity, demonstrating the ability to infer disease risk for future patients without requiring direct outcome labels. These findings highlight the potential of leveraging external bio-banks to extract valuable biomedical insights, even in data-limited scenarios. Our results demonstrate that machine learning models trained on smaller datasets can still be used to uncover real biological associations when carefully integrated with survival analysis and genetic studies. 

**Abstract (ZH)**: 机器学习在生物医学数据解释中的应用：即使在疾病标签有限的情况下也能揭示生物意义关联 

---
# MiZero: The Shadowy Defender Against Text Style Infringements 

**Title (ZH)**: MiZero: 阴影中的守护者——对抗文本风格侵权 

**Authors**: Ziwei Zhang, Juan Wen, Wanli Peng, Zhengxian Wu, Yinghan Zhou, Yiming Xue  

**Link**: [PDF](https://arxiv.org/pdf/2504.00035)  

**Abstract**: In-Context Learning (ICL) and efficient fine-tuning methods significantly enhanced the efficiency of applying Large Language Models (LLMs) to downstream tasks. However, they also raise concerns about the imitation and infringement of personal creative data. Current methods for data copyright protection primarily focuses on content security but lacks effectiveness in protecting the copyrights of text styles. In this paper, we introduce a novel implicit zero-watermarking scheme, namely MiZero. This scheme establishes a precise watermark domain to protect the copyrighted style, surpassing traditional watermarking methods that distort the style characteristics. Specifically, we employ LLMs to extract condensed-lists utilizing the designed instance delimitation mechanism. These lists guide MiZero in generating the watermark. Extensive experiments demonstrate that MiZero effectively verifies text style copyright ownership against AI imitation. 

**Abstract (ZH)**: 基于上下文学习(ICL)和高效微调方法显著提升了大型语言模型(LLMs)在下游任务中的应用效率。然而，它们也引发了对个人创作品复制和侵权的担忧。当前的数据版权保护方法主要集中在内容安全，但在保护文本风格版权方面效果有限。本文介绍了一种新颖的隐式零水印方案，即MiZero。该方案建立了精确的水印域，保护版权风格，超越了传统会扭曲风格特征的水印方法。具体来说，我们利用LLMs通过设计的实例分隔机制提取压缩列表，这些列表指导MiZero生成水印。广泛的经验实验表明，MiZero有效地验证了文本风格的版权归属，对抗AI模仿。 

---
# Leaking LoRa: An Evaluation of Password Leaks and Knowledge Storage in Large Language Models 

**Title (ZH)**: 泄露的LoRa：大规模语言模型中密码泄露和知识存储的评估 

**Authors**: Ryan Marinelli, Magnus Eckhoff  

**Link**: [PDF](https://arxiv.org/pdf/2504.00031)  

**Abstract**: To effectively deploy Large Language Models (LLMs) in application-specific settings, fine-tuning techniques are applied to enhance performance on specialized tasks. This process often involves fine-tuning on user data data, which may contain sensitive information. Although not recommended, it is not uncommon for users to send passwords in messages, and fine-tuning models on this could result in passwords being leaked. In this study, a Large Language Model is fine-tuned with customer support data and passwords from the RockYou password wordlist using Low-Rank Adaptation (LoRA). Out of the first 200 passwords from the list, 37 were successfully recovered. Further, causal tracing is used to identify that password information is largely located in a few layers. Lastly, Rank One Model Editing (ROME) is used to remove the password information from the model, resulting in the number of passwords recovered going from 37 to 0. 

**Abstract (ZH)**: 在特定应用场景中有效部署大规模语言模型：通过低秩适应（LoRA）微调客户支持数据和RockYou密码词表中的密码，去除密码信息以提升安全性 

---
# Token-Driven GammaTune: Adaptive Calibration for Enchanced Speculative Decoding 

**Title (ZH)**: 基于令牌驱动的 GammaTune：增强推测性解码的自适应校准 

**Authors**: Aayush Gautam, Susav Shrestha, Narasimha Annapareddy  

**Link**: [PDF](https://arxiv.org/pdf/2504.00030)  

**Abstract**: Speculative decoding accelerates large language model (LLM) inference by using a smaller draft model to propose tokens, which are then verified by a larger target model. However, selecting an optimal speculation length is critical for maximizing speedup while minimizing wasted computation. We introduce \textit{GammaTune} and \textit{GammaTune+}, training-free adaptive algorithms that dynamically adjust speculation length based on token acceptance rates using a heuristic-based switching mechanism. Evaluated on SpecBench across multiple tasks and model pairs, our method outperforms other heuristic-based approaches and fixed-length speculative decoding, achieving an average speedup of 15\% ($\pm$5\%) with \textit{GammaTune} and 16\% ($\pm$3\%) with \textit{GammaTune+}, while reducing performance variance. This makes \textit{GammaTune} a robust and efficient solution for real-world deployment. 

**Abstract (ZH)**: GammaTune 和 GammaTune+: 基于启发式动态调整猜想长度的无训练适应算法 

---
# Generating Structured Plan Representation of Procedures with LLMs 

**Title (ZH)**: 使用大语言模型生成程序的结构化计划表示 

**Authors**: Deepeka Garg, Sihan Zeng, Sumitra Ganesh, Leo Ardon  

**Link**: [PDF](https://arxiv.org/pdf/2504.00029)  

**Abstract**: In this paper, we address the challenges of managing Standard Operating Procedures (SOPs), which often suffer from inconsistencies in language, format, and execution, leading to operational inefficiencies. Traditional process modeling demands significant manual effort, domain expertise, and familiarity with complex languages like Business Process Modeling Notation (BPMN), creating barriers for non-techincal users. We introduce SOP Structuring (SOPStruct), a novel approach that leverages Large Language Models (LLMs) to transform SOPs into decision-tree-based structured representations. SOPStruct produces a standardized representation of SOPs across different domains, reduces cognitive load, and improves user comprehension by effectively capturing task dependencies and ensuring sequential integrity. Our approach enables leveraging the structured information to automate workflows as well as empower the human users. By organizing procedures into logical graphs, SOPStruct facilitates backtracking and error correction, offering a scalable solution for process optimization. We employ a novel evaluation framework, combining deterministic methods with the Planning Domain Definition Language (PDDL) to verify graph soundness, and non-deterministic assessment by an LLM to ensure completeness. We empirically validate the robustness of our LLM-based structured SOP representation methodology across SOPs from different domains and varying levels of complexity. Despite the current lack of automation readiness in many organizations, our research highlights the transformative potential of LLMs to streamline process modeling, paving the way for future advancements in automated procedure optimization. 

**Abstract (ZH)**: 本文介绍了管理标准操作程序（SOPs）所面临的挑战，由于SOPs在语言、格式和执行方面存在不一致性，导致操作效率低下。传统的流程建模需要大量的手动努力、领域专业知识以及熟悉复杂语言（如业务流程建模符号BPMN）的能力，这为非技术人员设置了障碍。我们提出了SOP结构化（SOPStruct）这一新颖的方法，利用大规模语言模型（LLMs）将SOPs转换为基于决策树的结构化表示。SOPStruct为不同领域的SOPs提供了标准化的表示形式，减少了认知负荷，通过有效捕捉任务依赖关系和确保顺序完整性，提高了用户理解能力。我们的方法能够利用结构化信息自动化工作流，同时增强人类用户的操作能力。通过将程序组织成逻辑图，SOPStruct支持回溯和错误修正，提供了一种可扩展的流程优化解决方案。我们采用了一种新的评估框架，结合确定性方法和规划领域定义语言（PDDL）验证图形的一致性，并通过LLM进行非确定性评估以确保完整性。我们通过跨不同领域和复杂程度的SOP的实证验证了我们基于LLM的结构化SOP表示方法的鲁棒性。尽管许多组织目前尚未准备好自动化，但我们的研究展示了LLMs在简化流程建模方面的变革潜力，为未来自动流程优化的进步铺平了道路。 

---
# Opioid Named Entity Recognition (ONER-2025) from Reddit 

**Title (ZH)**: Opioid Named Entity Recognition (ONER-2025) from Reddit 

**Authors**: Muhammad Ahmad, Humaira Farid, Iqra Ameer, Muhammad Muzamil, Ameer Hamza Muhammad Jalal, Ildar Batyrshin, Grigori Sidorov  

**Link**: [PDF](https://arxiv.org/pdf/2504.00027)  

**Abstract**: The opioid overdose epidemic remains a critical public health crisis, particularly in the United States, leading to significant mortality and societal costs. Social media platforms like Reddit provide vast amounts of unstructured data that offer insights into public perceptions, discussions, and experiences related to opioid use. This study leverages Natural Language Processing (NLP), specifically Opioid Named Entity Recognition (ONER-2025), to extract actionable information from these platforms. Our research makes four key contributions. First, we created a unique, manually annotated dataset sourced from Reddit, where users share self-reported experiences of opioid use via different administration routes. This dataset contains 331,285 tokens and includes eight major opioid entity categories. Second, we detail our annotation process and guidelines while discussing the challenges of labeling the ONER-2025 dataset. Third, we analyze key linguistic challenges, including slang, ambiguity, fragmented sentences, and emotionally charged language, in opioid discussions. Fourth, we propose a real-time monitoring system to process streaming data from social media, healthcare records, and emergency services to identify overdose events. Using 5-fold cross-validation in 11 experiments, our system integrates machine learning, deep learning, and transformer-based language models with advanced contextual embeddings to enhance understanding. Our transformer-based models (bert-base-NER and roberta-base) achieved 97% accuracy and F1-score, outperforming baselines by 10.23% (RF=0.88). 

**Abstract (ZH)**: opioid过量危机仍然是一个关键的公共卫生危机，特别是在美国，导致大量的死亡和社会成本。像Reddit这样的社交媒体平台提供了大量的非结构化数据，这些数据提供了关于公众对 opioids 使用的看法、讨论和经验的见解。本研究利用自然语言处理（NLP），特别是Opioid Named Entity Recognition (ONER-2025)，从这些平台中提取有用信息。我们的研究做出了四项关键贡献。首先，我们创建了一个独特的、人工标注的数据集，数据来源为Reddit，用户在此平台上分享不同给药途径的 opioid 自我报告使用经历。该数据集包含331,285个令牌，并包括八大主要 opioid 实体类别。其次，我们详细介绍了我们的标注过程和指南，同时讨论了标注ONER-2025数据集所面临的挑战。第三，我们分析了 opioid 讨论中的关键语言挑战，包括俚语、歧义、断裂的句子和情感化的语言。第四，我们提出了一种实时监测系统，用于处理来自社交媒体、医疗记录和紧急服务的流式数据，以识别过量事件。通过在11次实验中使用5折交叉验证，我们的系统结合了机器学习、深度学习和基于变换器的语言模型以及高级上下文嵌入，增强了对事件的理解。基于变换器的模型（bert-base-NER和roberta-base）实现了97%的准确率和F1分数，优于基线10.23%（RF=0.88）。 

---
# Diffusion models applied to skin and oral cancer classification 

**Title (ZH)**: 扩散模型在皮肤和口腔癌分类中的应用 

**Authors**: José J. M. Uliana, Renato A. Krohling  

**Link**: [PDF](https://arxiv.org/pdf/2504.00026)  

**Abstract**: This study investigates the application of diffusion models in medical image classification (DiffMIC), focusing on skin and oral lesions. Utilizing the datasets PAD-UFES-20 for skin cancer and P-NDB-UFES for oral cancer, the diffusion model demonstrated competitive performance compared to state-of-the-art deep learning models like Convolutional Neural Networks (CNNs) and Transformers. Specifically, for the PAD-UFES-20 dataset, the model achieved a balanced accuracy of 0.6457 for six-class classification and 0.8357 for binary classification (cancer vs. non-cancer). For the P-NDB-UFES dataset, it attained a balanced accuracy of 0.9050. These results suggest that diffusion models are viable models for classifying medical images of skin and oral lesions. In addition, we investigate the robustness of the model trained on PAD-UFES-20 for skin cancer but tested on the clinical images of the HIBA dataset. 

**Abstract (ZH)**: 该研究探讨了扩散模型在医学图像分类（DiffMIC）中的应用，重点关注皮肤和口腔病变。利用PAD-UFES-20数据集进行皮肤癌分类和P-NDB-UFES数据集进行口腔癌分类，扩散模型的性能与Convolutional Neural Networks (CNNs)和Transformers等先进深度学习模型相当。具体而言，PAD-UFES-20数据集上的六类分类实现了均衡精度0.6457，二类分类（癌症 vs. 非癌症）实现了均衡精度0.8357；P-NDB-UFES数据集上的均衡精度为0.9050。这些结果表明，扩散模型是分类皮肤和口腔病变医学图像的可行模型。此外，我们还研究了在PAD-UFES-20数据集上训练但在HIBA数据集的临床图像上测试的模型的鲁棒性。 

---
# A multi-locus predictiveness curve and its summary assessment for genetic risk prediction 

**Title (ZH)**: 多 locus 风险预测曲线及其遗传风险预测综合评估 

**Authors**: Changshuai Wei, Ming Li, Yalu Wen, Chengyin Ye, Qing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.00024)  

**Abstract**: With the advance of high-throughput genotyping and sequencing technologies, it becomes feasible to comprehensive evaluate the role of massive genetic predictors in disease prediction. There exists, therefore, a critical need for developing appropriate statistical measurements to access the combined effects of these genetic variants in disease prediction. Predictiveness curve is commonly used as a graphical tool to measure the predictive ability of a risk prediction model on a single continuous biomarker. Yet, for most complex diseases, risk prediciton models are formed on multiple genetic variants. We therefore propose a multi-marker predictiveness curve and provide a non-parametric method to construct the curve for case-control studies. We further introduce a global predictiveness U and a partial predictiveness U to summarize prediction curve across the whole population and sub-population of clinical interest, respectively. We also demonstrate the connections of predictiveness curve with ROC curve and Lorenz curve. Through simulation, we compared the performance of the predictiveness U to other three summary indices: R square, Total Gain, and Average Entropy, and showed that Predictiveness U outperformed the other three indexes in terms of unbiasedness and robustness. Moreover, we simulated a series of rare-variants disease model, found partial predictiveness U performed better than global predictiveness U. Finally, we conducted a real data analysis, using predictiveness curve and predictiveness U to evaluate a risk prediction model for Nicotine Dependence. 

**Abstract (ZH)**: 高通量基因分型和测序技术的发展使得全面评估大量遗传预测因子在疾病预测中的作用成为可能。因此，迫切需要开发适当的统计测量方法来评估这些遗传变异的综合效应对疾病预测的影响。预测曲线常被用作图形工具来衡量风险预测模型在单一连续生物标志物上的预测能力。然而，对于大多数复杂的疾病，风险预测模型是基于多个遗传变异形成的。因此，我们提出了一种多标记预测曲线，并提供了一种非参数方法来构建这种曲线，适用于病例对照研究。我们还引入了全局预测曲线U和部分预测曲线U来分别总结整个群体和临床兴趣子群体的预测曲线。我们还展示了预测曲线与ROC曲线和洛伦兹曲线之间的联系。通过模拟，我们将预测曲线U的性能与其他三个汇总指标（决定系数R平方、总增益和平均熵）进行了比较，并展示了预测曲线U在无偏性和稳健性方面的优越性。此外，我们模拟了一系列稀有变异疾病模型，发现部分预测曲线U在某些情况下优于全局预测曲线U。最后，我们进行了一项实际数据分析，使用预测曲线和预测曲线U来评估尼古丁依赖的风险预测模型。 

---
# Celler:A Genomic Language Model for Long-Tailed Single-Cell Annotation 

**Title (ZH)**: CELLER：一种针对长尾单细胞注释的基因组语言模型 

**Authors**: Huan Zhao, Yiming Liu, Jina Yao, Ling Xiong, Zexin Zhou, Zixing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00020)  

**Abstract**: Recent breakthroughs in single-cell technology have ushered in unparalleled opportunities to decode the molecular intricacy of intricate biological systems, especially those linked to diseases unique to humans. However, these progressions have also ushered in novel obstacles-specifically, the efficient annotation of extensive, long-tailed single-cell data pertaining to disease conditions. To effectively surmount this challenge, we introduce Celler, a state-of-the-art generative pre-training model crafted specifically for the annotation of single-cell data. Celler incorporates two groundbreaking elements: First, we introduced the Gaussian Inflation (GInf) Loss function. By dynamically adjusting sample weights, GInf Loss significantly enhances the model's ability to learn from rare categories while reducing the risk of overfitting for common categories. Secondly, we introduce an innovative Hard Data Mining (HDM) strategy into the training process, specifically targeting the challenging-to-learn minority data samples, which significantly improved the model's predictive accuracy. Additionally, to further advance research in this field, we have constructed a large-scale single-cell dataset: Celler-75, which encompasses 40 million cells distributed across 80 human tissues and 75 specific diseases. This dataset provides critical support for comprehensively exploring the potential of single-cell technology in disease research. Our code is available at this https URL. 

**Abstract (ZH)**: 近期单细胞技术的突破为解读复杂生物系统，尤其是与人类特有疾病相关的分子复杂性，带来了前所未有的机会。然而，这些进展也带来了一个新型挑战——高效标注广泛的长尾单细胞疾病数据。为有效克服这一挑战，我们引入了Celler，一种专为单细胞数据标注设计的先进生成预训练模型。Celler结合了两项创新元素：首先，我们引入了高斯膨胀损失函数（GInf Loss）。通过动态调整样本权重，GInf Loss显著增强了模型从稀有类别学习的能力，同时降低了对常见类别的过拟合风险。其次，我们在训练过程中引入了一种创新的难样本挖掘策略（HDM），专门针对难以学习的小众数据样本，显著提高了模型的预测准确性。此外，为了进一步推动该领域的研究，我们构建了一个大规模单细胞数据集——Celler-75，其中包括分布在80个人体组织和75种特定疾病中的4000万细胞。该数据集为全面探索单细胞技术在疾病研究中的潜力提供了关键支持。我们的代码可在以下链接获取：this https URL。 

---
# ObscuraCoder: Powering Efficient Code LM Pre-Training Via Obfuscation Grounding 

**Title (ZH)**: ObscuraCoder: 通过混淆基础实现高效代码LM预训练 

**Authors**: Indraneil Paul, Haoyi Yang, Goran Glavaš, Kristian Kersting, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2504.00019)  

**Abstract**: Language models (LMs) have become a staple of the code-writing toolbox. Their pre-training recipe has, however, remained stagnant over recent years, barring the occasional changes in data sourcing and filtering strategies. In particular, research exploring modifications to Code-LMs' pre-training objectives, geared towards improving data efficiency and better disentangling between syntax and semantics, has been noticeably sparse, especially compared with corresponding efforts in natural language LMs. In this work, we examine grounding on obfuscated code as a means of helping Code-LMs look beyond the surface-form syntax and enhance their pre-training sample efficiency. To this end, we compile ObscuraX, a dataset of approximately 55M source and obfuscated code pairs in seven languages. Subsequently, we pre-train ObscuraCoder models, ranging in size from 255M to 2.8B parameters, on a 272B-token corpus that includes ObscuraX and demonstrate that our obfuscation-based pre-training recipe leads to consistent improvements in Code-LMs' abilities compared to both vanilla autoregressive pre-training as well as existing de-obfuscation (DOBF) objectives. ObscuraCoder demonstrates sizeable gains across multiple tests of syntactic and semantic code understanding, along with improved capabilities in multilingual code completion, multilingual code commit summarization, and multi-purpose library-oriented code generation. 

**Abstract (ZH)**: 语言模型（LMs）已成为代码编写工具箱中的标准工具。然而，它们的预训练配方近年来尚未发生变化，仅偶尔在数据来源和筛选策略上有所改变。特别是针对代码-LMs的预训练目标进行改进的研究，以提高数据效率并更好地分离语法和语义的研究相对较少，尤其与自然语言LMs相应领域的努力相比。在这项工作中，我们研究将混淆代码作为手段，帮助代码-LMs超越表面语法结构，并提高其预训练样本效率。为此，我们编译了一个包含约5500万对源代码和混淆代码的ObscuraX数据集，涉及七种编程语言。随后，我们使用包含ObscuraX的272亿词语料库，对从2.55亿到28亿参数不等的ObscuraCoder模型进行了预训练，并证明基于混淆代码的预训练配方在与传统的自回归预训练以及现有的去混淆（DOBF）目标相比时，能够显著提高代码-LMs的能力。ObscuraCoder在多种语法和语义代码理解测试中表现出显著的进步，并且在多语言代码补全、多语言代码提交总结以及多用途面向库的代码生成方面也展现了增强的能力。 

---
# Enhance Vision-based Tactile Sensors via Dynamic Illumination and Image Fusion 

**Title (ZH)**: 通过动态照明和图像融合增强视觉触觉传感器 

**Authors**: Artemii Redkin, Zdravko Dugonjic, Mike Lambeta, Roberto Calandra  

**Link**: [PDF](https://arxiv.org/pdf/2504.00017)  

**Abstract**: Vision-based tactile sensors use structured light to measure deformation in their elastomeric interface. Until now, vision-based tactile sensors such as DIGIT and GelSight have been using a single, static pattern of structured light tuned to the specific form factor of the sensor. In this work, we investigate the effectiveness of dynamic illumination patterns, in conjunction with image fusion techniques, to improve the quality of sensing of vision-based tactile sensors. Specifically, we propose to capture multiple measurements, each with a different illumination pattern, and then fuse them together to obtain a single, higher-quality measurement. Experimental results demonstrate that this type of dynamic illumination yields significant improvements in image contrast, sharpness, and background difference. This discovery opens the possibility of retroactively improving the sensing quality of existing vision-based tactile sensors with a simple software update, and for new hardware designs capable of fully exploiting dynamic illumination. 

**Abstract (ZH)**: 基于视觉的触觉传感器使用结构光动态照明模式提高变形测量质量的研究 

---
# Deep Learning-Based Hypoglycemia Classification Across Multiple Prediction Horizons 

**Title (ZH)**: 基于深度学习的多预测 horizons 低血糖分类 

**Authors**: Beyza Cinar, Jennifer Daniel Onwuchekwa, Maria Maleshkova  

**Link**: [PDF](https://arxiv.org/pdf/2504.00009)  

**Abstract**: Type 1 diabetes (T1D) management can be significantly enhanced through the use of predictive machine learning (ML) algorithms, which can mitigate the risk of adverse events like hypoglycemia. Hypoglycemia, characterized by blood glucose levels below 70 mg/dL, is a life-threatening condition typically caused by excessive insulin administration, missed meals, or physical activity. Its asymptomatic nature impedes timely intervention, making ML models crucial for early detection. This study integrates short- (up to 2h) and long-term (up to 24h) prediction horizons (PHs) within a single classification model to enhance decision support. The predicted times are 5-15 min, 15-30 min, 30 min-1h, 1-2h, 2-4h, 4-8h, 8-12h, and 12-24h before hypoglycemia. In addition, a simplified model classifying up to 4h before hypoglycemia is compared. We trained ResNet and LSTM models on glucose levels, insulin doses, and acceleration data. The results demonstrate the superiority of the LSTM models when classifying nine classes. In particular, subject-specific models yielded better performance but achieved high recall only for classes 0, 1, and 2 with 98%, 72%, and 50%, respectively. A population-based six-class model improved the results with at least 60% of events detected. In contrast, longer PHs remain challenging with the current approach and may be considered with different models. 

**Abstract (ZH)**: 通过使用预测机器学习算法，1型糖尿病管理可以显著增强，从而减轻低血糖等不良事件的风险。 

---
# Tensor Generalized Approximate Message Passing 

**Title (ZH)**: 张量广义消息传递逼近 

**Authors**: Yinchuan Li, Guangchen Lan, Xiaodong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00008)  

**Abstract**: We propose a tensor generalized approximate message passing (TeG-AMP) algorithm for low-rank tensor inference, which can be used to solve tensor completion and decomposition problems. We derive TeG-AMP algorithm as an approximation of the sum-product belief propagation algorithm in high dimensions where the central limit theorem and Taylor series approximations are applicable. As TeG-AMP is developed based on a general TR decomposition model, it can be directly applied to many low-rank tensor types. Moreover, our TeG-AMP can be simplified based on the CP decomposition model and a tensor simplified AMP is proposed for low CP-rank tensor inference problems. Experimental results demonstrate that the proposed methods significantly improve recovery performances since it takes full advantage of tensor structures. 

**Abstract (ZH)**: 我们提出了一种张量广义消息传递（TeG-AMP）算法用于低秩张量推理，可以用于解决张量填充和分解问题。 

---
# Are We There Yet? A Measurement Study of Efficiency for LLM Applications on Mobile Devices 

**Title (ZH)**: 我们到了吗？移动设备上大语言模型应用的效率测量研究 

**Authors**: Xiao Yan, Yi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2504.00002)  

**Abstract**: Recent advancements in large language models (LLMs) have prompted interest in deploying these models on mobile devices to enable new applications without relying on cloud connectivity. However, the efficiency constraints of deploying LLMs on resource-limited devices present significant challenges. In this paper, we conduct a comprehensive measurement study to evaluate the efficiency tradeoffs between mobile-based, edge-based, and cloud-based deployments for LLM applications. We implement AutoLife-Lite, a simplified LLM-based application that analyzes smartphone sensor data to infer user location and activity contexts. Our experiments reveal that: (1) Only small-size LLMs (<4B parameters) can run successfully on powerful mobile devices, though they exhibit quality limitations compared to larger models; (2) Model compression is effective in lower the hardware requirement, but may lead to significant performance degradation; (3) The latency to run LLMs on mobile devices with meaningful output is significant (>30 seconds), while cloud services demonstrate better time efficiency (<10 seconds); (4) Edge deployments offer intermediate tradeoffs between latency and model capabilities, with different results on CPU-based and GPU-based settings. These findings provide valuable insights for system designers on the current limitations and future directions for on-device LLM applications. 

**Abstract (ZH)**: 近期大规模语言模型的进步激发了在移动设备上部署这些模型的兴趣，以在无需依靠云连接的情况下启用新的应用程序。然而，资源受限设备上部署大规模语言模型的效率限制带来了重大挑战。本文开展了一项全面的测量研究，评估了在移动设备、边缘设备和云端部署大规模语言模型应用之间的效率权衡。我们实现了AutoLife-Lite，这是一个简化的大规模语言模型应用，分析智能手机传感器数据以推断用户的位置和活动背景。实验结果显示：（1）只有参数量小于4B的小型语言模型能在强大的移动设备上成功运行，但其在质量上存在局限性，相比大型模型；（2）模型压缩有效降低了硬件需求，但可能导致性能显著下降；（3）使用移动设备运行大规模语言模型以获得有意义输出的延迟显著（>30秒），而云服务表现出更好的时间效率（<10秒）；（4）边缘部署在延迟和模型能力之间提供了中间权衡，不同结果分别在基于CPU和基于GPU的设置下。这些发现为系统设计者提供了关于当前限制和未来方向的重要见解，以支持设备端的大规模语言模型应用。 

---
