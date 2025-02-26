# MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning 

**Title (ZH)**: MAPoRL: 多智能体共训练后微调促进合作型大规模语言模型的强化学习方法 

**Authors**: Chanwoo Park, Seungju Han, Xingzhi Guo, Asuman Ozdaglar, Kaiqing Zhang, Joo-Kyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.18439)  

**Abstract**: Leveraging multiple large language models (LLMs) to build collaborative multi-agentic workflows has demonstrated significant potential. However, most previous studies focus on prompting the out-of-the-box LLMs, relying on their innate capability for collaboration, which may not improve LLMs' performance as shown recently. In this paper, we introduce a new post-training paradigm MAPoRL (Multi-Agent Post-co-training for collaborative LLMs with Reinforcement Learning), to explicitly elicit the collaborative behaviors and further unleash the power of multi-agentic LLM frameworks. In MAPoRL, multiple LLMs first generate their own responses independently and engage in a multi-turn discussion to collaboratively improve the final answer. In the end, a MAPoRL verifier evaluates both the answer and the discussion, by assigning a score that verifies the correctness of the answer, while adding incentives to encourage corrective and persuasive discussions. The score serves as the co-training reward, and is then maximized through multi-agent RL. Unlike existing LLM post-training paradigms, MAPoRL advocates the co-training of multiple LLMs together using RL for better generalization. Accompanied by analytical insights, our experiments demonstrate that training individual LLMs alone is insufficient to induce effective collaboration. In contrast, multi-agent co-training can boost the collaboration performance across benchmarks, with generalization to unseen domains. 

**Abstract (ZH)**: 利用多个大型语言模型（LLMs）构建协作多代理工作流表现出显著潜力。然而，大多数前期研究主要关注通过提示开箱即用的LLMs，依赖其固有的协作能力，这可能并未如近期所显示的那样改善LLMs的表现。本文介绍了一种新的后训练范式MAPoRL（多代理后共训练以强化学习促进协同LLMs），以明确激发协同行为并进一步释放多代理LLMs框架的力量。在MAPoRL中，多个LLMs首先独立生成自己的响应，并进行多轮讨论以合作提高最终答案。最后，一个MAPoRL验证器通过分配验证答案正确性的分数来评估答案和讨论，同时通过鼓励纠正性和说服性的讨论来增强激励机制。该分数作为共训练奖励，并通过多代理强化学习进行最大化。与现有的LLM后训练范式不同，MAPoRL提倡使用强化学习共同训练多个LLMs以获得更好的泛化能力。我们的实验分析表明，单独训练单个LLM不足以引发有效的协作。相比之下，多代理共训练可以在各种基准测试中提升协作性能，并扩展到未见过的领域。 

---
# PyEvalAI: AI-assisted evaluation of Jupyter Notebooks for immediate personalized feedback 

**Title (ZH)**: PyEvalAI：AI辅助的Jupyter Notebook评估以提供即时个性化反馈 

**Authors**: Nils Wandel, David Stotko, Alexander Schier, Reinhard Klein  

**Link**: [PDF](https://arxiv.org/pdf/2502.18425)  

**Abstract**: Grading student assignments in STEM courses is a laborious and repetitive task for tutors, often requiring a week to assess an entire class. For students, this delay of feedback prevents iterating on incorrect solutions, hampers learning, and increases stress when exercise scores determine admission to the final exam. Recent advances in AI-assisted education, such as automated grading and tutoring systems, aim to address these challenges by providing immediate feedback and reducing grading workload. However, existing solutions often fall short due to privacy concerns, reliance on proprietary closed-source models, lack of support for combining Markdown, LaTeX and Python code, or excluding course tutors from the grading process. To overcome these limitations, we introduce PyEvalAI, an AI-assisted evaluation system, which automatically scores Jupyter notebooks using a combination of unit tests and a locally hosted language model to preserve privacy. Our approach is free, open-source, and ensures tutors maintain full control over the grading process. A case study demonstrates its effectiveness in improving feedback speed and grading efficiency for exercises in a university-level course on numerics. 

**Abstract (ZH)**: STEM课程作业评分是导师的一项耗费时间和重复的工作，通常评估一个班级需要一周时间。对于学生来说，这种反馈延迟阻碍了他们对错误解决方案的迭代，影响学习，并在练习成绩决定是否能参加最终考试时增加压力。近年来，辅助教育的AI技术，如自动化评分和辅导系统，旨在通过提供即时反馈和减少评分工作量来解决这些挑战。然而，现有的解决方案往往因为隐私问题、依赖专有封闭源模型、不支持结合Markdown、LaTeX和Python代码或不包括课程导师在评分过程中等原因而存在不足。为克服这些限制，我们引入了PyEvalAI，这是一种AI辅助评估系统，通过结合单元测试和本地托管的语言模型自动评分，以保护隐私。我们的方法是免费开源的，并确保导师能够完全控制评分过程。案例研究证明了其在大学数值课程中提高练习反馈速度和评分效率方面的有效性。 

---
# The Gradient of Algebraic Model Counting 

**Title (ZH)**: 代数模型计数的梯度 

**Authors**: Jaron Maene, Luc De Raedt  

**Link**: [PDF](https://arxiv.org/pdf/2502.18406)  

**Abstract**: Algebraic model counting unifies many inference tasks on logic formulas by exploiting semirings. Rather than focusing on inference, we consider learning, especially in statistical-relational and neurosymbolic AI, which combine logical, probabilistic and neural representations. Concretely, we show that the very same semiring perspective of algebraic model counting also applies to learning. This allows us to unify various learning algorithms by generalizing gradients and backpropagation to different semirings. Furthermore, we show how cancellation and ordering properties of a semiring can be exploited for more memory-efficient backpropagation. This allows us to obtain some interesting variations of state-of-the-art gradient-based optimisation methods for probabilistic logical models. We also discuss why algebraic model counting on tractable circuits does not lead to more efficient second-order optimization. Empirically, our algebraic backpropagation exhibits considerable speed-ups as compared to existing approaches. 

**Abstract (ZH)**: 代数模型计数通过利用半环统一了许多逻辑公式上的推理任务。不同于关注推理，我们探讨学习，特别是在统计关系和神经符号AI中，这些领域结合了逻辑、概率和神经表示。具体而言，我们展示了相同的半环视角也适用于学习。这使我们可以通过将梯度和反向传播一般化到不同的半环来统一各种学习算法。此外，我们展示了如何利用半环的取消和排序性质以更高效地进行反向传播。这使我们能够获得一些有趣的状态-of-the-art概率逻辑模型梯度优化方法的变体。我们还讨论了为什么在可处理电路上的代数模型计数不会导致更高效的二阶优化。实验上，我们的代数反向传播相比现有方法表现出显著的速度提升。 

---
# How Far are LLMs from Real Search? A Comprehensive Study on Efficiency, Completeness, and Inherent Capabilities 

**Title (ZH)**: 大型语言模型与实际搜索相差多远？关于效率、完备性及内在能力的全面研究 

**Authors**: Minhua Lin, Hui Liu, Xianfeng Tang, Jingying Zeng, Zhenwei Dai, Chen Luo, Zheng Li, Xiang Zhang, Qi He, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18387)  

**Abstract**: Search plays a fundamental role in problem-solving across various domains, with most real-world decision-making problems being solvable through systematic search. Drawing inspiration from recent discussions on search and learning, we systematically explore the complementary relationship between search and Large Language Models (LLMs) from three perspectives. First, we analyze how learning can enhance search efficiency and propose Search via Learning (SeaL), a framework that leverages LLMs for effective and efficient search. Second, we further extend SeaL to SeaL-C to ensure rigorous completeness during search. Our evaluation across three real-world planning tasks demonstrates that SeaL achieves near-perfect accuracy while reducing search spaces by up to 99.1% compared to traditional approaches. Finally, we explore how far LLMs are from real search by investigating whether they can develop search capabilities independently. Our analysis reveals that while current LLMs struggle with efficient search in complex problems, incorporating systematic search strategies significantly enhances their problem-solving capabilities. These findings not only validate the effectiveness of our approach but also highlight the need for improving LLMs' search abilities for real-world applications. 

**Abstract (ZH)**: 搜索在各个领域的问题解决中起着基础性作用，大多数现实世界中的决策问题可以通过系统的搜索方法来解决。受到搜索与学习近期讨论的启发，我们从三个方面系统地探讨了搜索与大型语言模型（LLMs）之间的互补关系。首先，我们分析了学习如何提升搜索效率，并提出了基于LLMs的搜索框架Search via Learning (SeaL)，以实现高效搜索。其次，我们将SeaL扩展为SeaL-C，以确保搜索过程的严格完备性。我们的评估结果显示，SeaL在三项真实世界规划任务中的准确率达到接近完美，并将搜索空间减少了高达99.1%。最后，我们探讨了当前LLMs在独立开发搜索能力方面的局限性，并分析了系统化搜索策略如何显著增强其解决问题的能力。这些发现不仅验证了我们方法的有效性，还强调了提高LLMs搜索能力以适应现实世界应用的必要性。 

---
# MindMem: Multimodal for Predicting Advertisement Memorability Using LLMs and Deep Learning 

**Title (ZH)**: MindMem：多模态预测广告记忆性的人工智能与深度学习方法 

**Authors**: Sepehr Asgarian, Qayam Jetha, Jouhyun Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2502.18371)  

**Abstract**: In the competitive landscape of advertising, success hinges on effectively navigating and leveraging complex interactions among consumers, advertisers, and advertisement platforms. These multifaceted interactions compel advertisers to optimize strategies for modeling consumer behavior, enhancing brand recall, and tailoring advertisement content. To address these challenges, we present MindMem, a multimodal predictive model for advertisement memorability. By integrating textual, visual, and auditory data, MindMem achieves state-of-the-art performance, with a Spearman's correlation coefficient of 0.631 on the LAMBDA and 0.731 on the Memento10K dataset, consistently surpassing existing methods. Furthermore, our analysis identified key factors influencing advertisement memorability, such as video pacing, scene complexity, and emotional resonance. Expanding on this, we introduced MindMem-ReAd (MindMem-Driven Re-generated Advertisement), which employs Large Language Model-based simulations to optimize advertisement content and placement, resulting in up to a 74.12% improvement in advertisement memorability. Our results highlight the transformative potential of Artificial Intelligence in advertising, offering advertisers a robust tool to drive engagement, enhance competitiveness, and maximize impact in a rapidly evolving market. 

**Abstract (ZH)**: 在广告竞争 landscape 中，成功取决于有效地导航和利用消费者、广告商和广告平台之间错综复杂的相互作用。这些多维度的相互作用促使广告商优化策略以建模消费者行为、增强品牌回忆并定制广告内容。为应对这些挑战，我们提出了 MindMem，一种多模态预测模型，用于广告记忆性。通过整合文本、视觉和音频数据，MindMem 实现了最先进的性能，其在 LAMBDA 数据集上的 Spearman 相关系数为 0.631，在 Memento10K 数据集上的相关系数为 0.731，始终超越现有方法。此外，我们的分析确定了影响广告记忆性的关键因素，如视频节奏、场景复杂性和情感共鸣。在此基础上，我们引入了 MindMem-ReAd（由 MindMem 驱动的重新生成广告），利用基于大语言模型的模拟优化广告内容和投放，广告记忆提升高达 74.12%。我们的结果突显了人工智能在广告领域的变革潜力，为广告商提供了一种强大的工具，以推动参与度、增强竞争力并最大化在快速变化市场中的影响力。 

---
# GraphRank Pro+: Advancing Talent Analytics Through Knowledge Graphs and Sentiment-Enhanced Skill Profiling 

**Title (ZH)**: GraphRank Pro+: 通过知识图谱和情绪增强技能画像推动人才分析advance 

**Authors**: Sirisha Velampalli, Chandrashekar Muniyappa  

**Link**: [PDF](https://arxiv.org/pdf/2502.18315)  

**Abstract**: The extraction of information from semi-structured text, such as resumes, has long been a challenge due to the diverse formatting styles and subjective content organization. Conventional solutions rely on specialized logic tailored for specific use cases. However, we propose a revolutionary approach leveraging structured Graphs, Natural Language Processing (NLP), and Deep Learning. By abstracting intricate logic into Graph structures, we transform raw data into a comprehensive Knowledge Graph. This innovative framework enables precise information extraction and sophisticated querying. We systematically construct dictionaries assigning skill weights, paving the way for nuanced talent analysis. Our system not only benefits job recruiters and curriculum designers but also empowers job seekers with targeted query-based filtering and ranking capabilities. 

**Abstract (ZH)**: 从半结构化文本中提取信息由于多样化的格式风格和主观的内容组织长期是一项挑战。传统解决方案依赖于针对特定用途定制的专业逻辑。然而，我们提出了一种革命性的方法，利用结构化图、自然语言处理（NLP）和深度学习。通过将复杂的逻辑抽象为图结构，我们将原始数据转换为全面的知识图谱。这一创新框架使精确的信息抽取和复杂的查询成为可能。我们系统地构建词典分配技能权重，为精细的人才分析铺平道路。该系统不仅有利于招聘人员和课程设计师，还为求职者提供了基于查询的过滤和排名能力。 

---
# Citrus: Leveraging Expert Cognitive Pathways in a Medical Language Model for Advanced Medical Decision Support 

**Title (ZH)**: 柑橘：在医疗语言模型中利用专家认知路径以实现高级医疗决策支持 

**Authors**: Guoxin Wang, Minyu Gao, Shuai Yang, Ya Zhang, Lizhi He, Liang Huang, Hanlin Xiao, Yexuan Zhang, Wanyue Li, Lu Chen, Jintao Fei, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.18274)  

**Abstract**: Large language models (LLMs), particularly those with reasoning capabilities, have rapidly advanced in recent years, demonstrating significant potential across a wide range of applications. However, their deployment in healthcare, especially in disease reasoning tasks, is hindered by the challenge of acquiring expert-level cognitive data. In this paper, we introduce Citrus, a medical language model that bridges the gap between clinical expertise and AI reasoning by emulating the cognitive processes of medical experts. The model is trained on a large corpus of simulated expert disease reasoning data, synthesized using a novel approach that accurately captures the decision-making pathways of clinicians. This approach enables Citrus to better simulate the complex reasoning processes involved in diagnosing and treating medical this http URL further address the lack of publicly available datasets for medical reasoning tasks, we release the last-stage training data, including a custom-built medical diagnostic dialogue dataset. This open-source contribution aims to support further research and development in the field. Evaluations using authoritative benchmarks such as MedQA, covering tasks in medical reasoning and language understanding, show that Citrus achieves superior performance compared to other models of similar size. These results highlight Citrus potential to significantly enhance medical decision support systems, providing a more accurate and efficient tool for clinical decision-making. 

**Abstract (ZH)**: 大型语言模型（LLMs），特别是那些具有推理能力的模型，在近年来迅速发展，展示了在广泛的应用领域中的巨大潜力。然而，它们在医疗领域的部署，尤其是在疾病推理任务中的应用，受限于难以获取专家级的认知数据。本文介绍了一种名为Citrus的医疗语言模型，通过模拟医疗专家的认知过程，填补了临床专业知识和AI推理之间的差距。模型是在一个大型的模拟专家疾病推理数据集上进行训练的，该数据集采用了新颖的方法合成，准确地捕捉了临床人员的决策路径。这种做法使得Citrus能够更好地模拟诊断和治疗过程中所涉及的复杂推理过程。为了解决医疗推理任务中缺乏公开数据集的问题，我们发布了训练数据的最终阶段，包括一个自建的医疗诊断对话数据集。这一开源贡献旨在支持该领域的进一步研究和开发。使用权威基准如MedQA进行的评估涵盖医疗推理和语言理解任务，结果显示Citrus在性能上优于其他同类大小的模型。这些结果突显了Citrus在显著增强医疗决策支持系统方面的重要性，提供了一个更准确和高效的临床决策工具。 

---
# ChatMotion: A Multimodal Multi-Agent for Human Motion Analysis 

**Title (ZH)**: ChatMotion：多模态多代理human motion分析 

**Authors**: Li Lei, Jia Sen, Wang Jianhao, An Zhaochong, Li Jiaang, Hwang Jenq-Neng, Belongie Serge  

**Link**: [PDF](https://arxiv.org/pdf/2502.18180)  

**Abstract**: Advancements in Multimodal Large Language Models (MLLMs) have improved human motion understanding. However, these models remain constrained by their "instruct-only" nature, lacking interactivity and adaptability for diverse analytical perspectives. To address these challenges, we introduce ChatMotion, a multimodal multi-agent framework for human motion analysis. ChatMotion dynamically interprets user intent, decomposes complex tasks into meta-tasks, and activates specialized function modules for motion comprehension. It integrates multiple specialized modules, such as the MotionCore, to analyze human motion from various perspectives. Extensive experiments demonstrate ChatMotion's precision, adaptability, and user engagement for human motion understanding. 

**Abstract (ZH)**: multimodal 多模态大型语言模型（MLLMs）的进步提高了对人类运动的理解。然而，这些模型仍然受到“仅指令”性质的限制，缺乏互动性和多样性分析视角的适应性。为应对这些挑战，我们提出了 ChatMotion，一种用于人类运动分析的多模态多agent框架。ChatMotion动态解读用户意图，将复杂任务分解为元任务，并激活专门的功能模块以进行运动理解。它集成了多个专门模块，如MotionCore，从多种视角分析人类运动。大量实验展示了ChatMotion在人类运动理解方面的精确性、适应性和用户参与度。 

---
# Defining bias in AI-systems: Biased models are fair models 

**Title (ZH)**: 定义AI系统中的偏见：有偏见的模型即公平的模型 

**Authors**: Chiara Lindloff, Ingo Siegert  

**Link**: [PDF](https://arxiv.org/pdf/2502.18060)  

**Abstract**: The debate around bias in AI systems is central to discussions on algorithmic fairness. However, the term bias often lacks a clear definition, despite frequently being contrasted with fairness, implying that an unbiased model is inherently fair. In this paper, we challenge this assumption and argue that a precise conceptualization of bias is necessary to effectively address fairness concerns. Rather than viewing bias as inherently negative or unfair, we highlight the importance of distinguishing between bias and discrimination. We further explore how this shift in focus can foster a more constructive discourse within academic debates on fairness in AI systems. 

**Abstract (ZH)**: AI系统中的偏差辩论是算法公平性讨论中的核心议题。然而，偏差一词往往缺乏明确的定义，尽管它常被与公平性对立使用，暗示无偏见的模型必然是公平的。本文挑战这一假设，认为需要对偏差进行精确的概念化以便有效应对公平性问题。我们强调区分偏差与歧视的重要性，而不是将偏差视为固有的负面或不公平。此外，我们探讨了这种焦点转移如何促进学术界对AI系统公平性的建设性讨论。 

---
# GNN-XAR: A Graph Neural Network for Explainable Activity Recognition in Smart Homes 

**Title (ZH)**: GNN-XAR：一种可解释的家庭智能活动识别图神经网络 

**Authors**: Michele Fiori, Davide Mor, Gabriele Civitarese, Claudio Bettini  

**Link**: [PDF](https://arxiv.org/pdf/2502.17999)  

**Abstract**: Sensor-based Human Activity Recognition (HAR) in smart home environments is crucial for several applications, especially in the healthcare domain. The majority of the existing approaches leverage deep learning models. While these approaches are effective, the rationale behind their outputs is opaque. Recently, eXplainable Artificial Intelligence (XAI) approaches emerged to provide intuitive explanations to the output of HAR models. To the best of our knowledge, these approaches leverage classic deep models like CNNs or RNNs. Recently, Graph Neural Networks (GNNs) proved to be effective for sensor-based HAR. However, existing approaches are not designed with explainability in mind. In this work, we propose the first explainable Graph Neural Network explicitly designed for smart home HAR. Our results on two public datasets show that this approach provides better explanations than state-of-the-art methods while also slightly improving the recognition rate. 

**Abstract (ZH)**: 基于传感器的智能家居环境中的人体活动识别（HAR）在多个应用中至关重要，尤其是在医疗健康领域。现有的大多数方法依赖深度学习模型。尽管这些方法有效，但它们的输出 rationale 难以理解。最近，可解释人工智能（XAI）方法出现，旨在提供直观的解释以供HAR模型使用。据我们所知，这些方法主要依赖于经典的深度模型，如CNN或RNN。最近，图形神经网络（GNN）证明了对于基于传感器的HAR的有效性。然而，现有的方法并未考虑可解释性。在本工作中，我们提出了第一个专门为智能家居HAR设计的可解释图形神经网络。我们在两个公共数据集上的结果表明，该方法不仅提供了比现有最佳方法更好的解释，还在一定程度上提高了识别率。 

---
# LeanProgress: Guiding Search for Neural Theorem Proving via Proof Progress Prediction 

**Title (ZH)**: LeanProgress: 通过证明进度预测指导神经定理证明搜索 

**Authors**: Suozhi Huang, Peiyang Song, Robert Joseph George, Anima Anandkumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.17925)  

**Abstract**: Mathematical reasoning remains a significant challenge for Large Language Models (LLMs) due to hallucinations. When combined with formal proof assistants like Lean, these hallucinations can be eliminated through rigorous verification, making theorem proving reliable. However, even with formal verification, LLMs still struggle with long proofs and complex mathematical formalizations. While Lean with LLMs offers valuable assistance with retrieving lemmas, generating tactics, or even complete proofs, it lacks a crucial capability: providing a sense of proof progress. This limitation particularly impacts the overall development efficiency in large formalization projects. We introduce LeanProgress, a method that predicts the progress in the proof. Training and evaluating our models made on a large corpus of Lean proofs from Lean Workbook Plus and Mathlib4 and how many steps remain to complete it, we employ data preprocessing and balancing techniques to handle the skewed distribution of proof lengths. Our experiments show that LeanProgress achieves an overall prediction accuracy of 75.1\% in predicting the amount of progress and, hence, the remaining number of steps. When integrated into a best-first search framework using Reprover, our method shows a 3.8\% improvement on Mathlib4 compared to baseline performances of 41.2\%, particularly for longer proofs. These results demonstrate how proof progress prediction can enhance both automated and interactive theorem proving, enabling users to make more informed decisions about proof strategies. 

**Abstract (ZH)**: Large Language Models (LLMs)中的数学推理仍因幻觉而构成重大挑战。结合形式证明助手Lean后，这些幻觉可以通过严格的验证消除，从而使定理证明变得可靠。然而，即使经过形式验证，LLMs依然难以处理长证明和复杂的数学形式化。虽然Lean与LLMs结合可以为检索引理、生成策略或甚至完整证明提供有价值的帮助，但它缺乏一个关键能力：提供证明进度感。这一限制尤其影响大规模形式化项目的整体开发效率。我们介绍了LeanProgress，这是一种预测证明进度的方法。我们在Lean Workbook Plus和Mathlib4的大规模Lean证明语料库上训练和评估了我们的模型，预测剩余步骤的数量，并采用数据预处理和平衡技术来处理证明长度的偏斜分布。我们的实验结果显示，LeanProgress在预测证明进度和剩余步骤数量方面总体准确率为75.1%。将其集成到使用Reprover的最佳首先搜索框架中时，与基准性能41.2%相比，我们的方法在Mathlib4上显示出3.8%的改进，特别是在较长的证明中。这些结果表明，证明进度预测可以增强自动化的和交互式的定理证明，使用户能够就证明策略做出更加明智的决策。 

---
# Unmasking Gender Bias in Recommendation Systems and Enhancing Category-Aware Fairness 

**Title (ZH)**: 揭示推荐系统中的性别偏见并增强类别意识公平性 

**Authors**: Tahsin Alamgir Kheya, Mohamed Reda Bouadjenek, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2502.17921)  

**Abstract**: Recommendation systems are now an integral part of our daily lives. We rely on them for tasks such as discovering new movies, finding friends on social media, and connecting job seekers with relevant opportunities. Given their vital role, we must ensure these recommendations are free from societal stereotypes. Therefore, evaluating and addressing such biases in recommendation systems is crucial. Previous work evaluating the fairness of recommended items fails to capture certain nuances as they mainly focus on comparing performance metrics for different sensitive groups. In this paper, we introduce a set of comprehensive metrics for quantifying gender bias in recommendations. Specifically, we show the importance of evaluating fairness on a more granular level, which can be achieved using our metrics to capture gender bias using categories of recommended items like genres for movies. Furthermore, we show that employing a category-aware fairness metric as a regularization term along with the main recommendation loss during training can help effectively minimize bias in the models' output. We experiment on three real-world datasets, using five baseline models alongside two popular fairness-aware models, to show the effectiveness of our metrics in evaluating gender bias. Our metrics help provide an enhanced insight into bias in recommended items compared to previous metrics. Additionally, our results demonstrate how incorporating our regularization term significantly improves the fairness in recommendations for different categories without substantial degradation in overall recommendation performance. 

**Abstract (ZH)**: 推荐系统中的性别偏见量化及其缓解方法 

---
# Towards Sustainable Web Agents: A Plea for Transparency and Dedicated Metrics for Energy Consumption 

**Title (ZH)**: 面向可持续的网络代理：呼吁透明度和专门的能源消耗指标 

**Authors**: Lars Krupp, Daniel Geißler, Paul Lukowicz, Jakob Karolus  

**Link**: [PDF](https://arxiv.org/pdf/2502.17903)  

**Abstract**: Improvements in the area of large language models have shifted towards the construction of models capable of using external tools and interpreting their outputs. These so-called web agents have the ability to interact autonomously with the internet. This allows them to become powerful daily assistants handling time-consuming, repetitive tasks while supporting users in their daily activities. While web agent research is thriving, the sustainability aspect of this research direction remains largely unexplored. We provide an initial exploration of the energy and CO2 cost associated with web agents. Our results show how different philosophies in web agent creation can severely impact the associated expended energy. We highlight lacking transparency regarding the disclosure of model parameters and processes used for some web agents as a limiting factor when estimating energy consumption. As such, our work advocates a change in thinking when evaluating web agents, warranting dedicated metrics for energy consumption and sustainability. 

**Abstract (ZH)**: 大型语言模型领域的改进已转向构建能够使用外部工具并解释其输出的模型。这些所谓的网络代理具备自主与互联网交互的能力。这使得它们能够成为处理耗时且重复的任务的强大日常助手，同时支持用户的日常活动。尽管网络代理研究蓬勃发展，但这一研究方向的可持续性方面仍鲜有探索。我们对网络代理相关的能源及二氧化碳成本进行了初步探索。结果显示，网络代理创建中不同的哲学观点可能导致显著的能源消耗差异。此外，我们指出，在估算能源消耗时，一些网络代理缺乏透明度，未披露模型参数和过程是一个限制因素。因此，我们的研究倡导在评估网络代理时改变思维方式，需要专门的能源消耗和可持续性指标。 

---
# Science Across Languages: Assessing LLM Multilingual Translation of Scientific Papers 

**Title (ZH)**: 跨语言科学：评估LLM在多语言翻译科学论文中的表现 

**Authors**: Hannah Calzi Kleidermacher, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2502.17882)  

**Abstract**: Scientific research is inherently global. However, the vast majority of academic journals are published exclusively in English, creating barriers for non-native-English-speaking researchers. In this study, we leverage large language models (LLMs) to translate published scientific articles while preserving their native JATS XML formatting, thereby developing a practical, automated approach for implementation by academic journals. Using our approach, we translate articles across multiple scientific disciplines into 28 languages. To evaluate translation accuracy, we introduce a novel question-and-answer (QA) benchmarking method, in which an LLM generates comprehension-based questions from the original text and then answers them based on the translated text. Our benchmark results show an average performance of 95.9%, showing that the key scientific details are accurately conveyed. In a user study, we translate the scientific papers of 15 researchers into their native languages, finding that the authors consistently found the translations to accurately capture the original information in their articles. Interestingly, a third of the authors found many technical terms "overtranslated," expressing a preference to keep terminology more familiar in English untranslated. Finally, we demonstrate how in-context learning techniques can be used to align translations with domain-specific preferences such as mitigating overtranslation, highlighting the adaptability and utility of LLM-driven scientific translation. The code and translated articles are available at this https URL. 

**Abstract (ZH)**: 科学研究本质上是全球性的。然而，绝大多数学术期刊仅以英文出版，为非英语母语的研究者设置了障碍。本研究利用大规模语言模型（LLMs）翻译已发表的科学文章，同时保留其原生的JATS XML格式，从而开发了一种适用于学术期刊的实用自动化方法。使用本方法，我们将多学科的论文翻译成28种语言。为了评估翻译准确性，我们引入了一种新颖的问答（QA）基准测试方法，其中大规模语言模型从原文生成基于理解的问题，并基于译文作答。基准测试结果表明平均正确率为95.9%，表明关键科学细节得到了准确传达。在一项用户研究中，我们将15位研究人员的科学论文翻译成他们的母语，发现作者一致认为翻译准确捕捉了原文信息。有趣的是，三分之一的作者发现许多技术术语被过度翻译，更倾向于保留一些术语不翻译。最后，我们展示了如何使用上下文学习技术对翻译与学科特定偏好进行对齐，例如减少过度翻译，突显了基于大规模语言模型的科学翻译的适应性和实用性。代码和翻译的文章可在以下网址获取。 

---
# A Combinatorial Identities Benchmark for Theorem Proving via Automated Theorem Generation 

**Title (ZH)**: 一个组合恒等式基准测试集，用于通过自动定理生成进行定理证明。 

**Authors**: Beibei Xiong, Hangyu Lv, Haojia Shan, Jianlin Wang, Zhengfeng Yang, Lihong Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2502.17840)  

**Abstract**: Large language models (LLMs) have significantly advanced formal theorem proving, yet the scarcity of high-quality training data constrains their capabilities in complex mathematical domains. Combinatorics, a cornerstone of mathematics, provides essential tools for analyzing discrete structures and solving optimization problems. However, its inherent complexity makes it particularly challenging for automated theorem proving (ATP) for combinatorial identities. To address this, we manually construct LeanComb, combinatorial identities benchmark in Lean, which is, to our knowledge, the first formalized theorem proving benchmark built for combinatorial identities. We develop an Automated Theorem Generator for Combinatorial Identities, ATG4CI, which combines candidate tactics suggested by a self-improving large language model with a Reinforcement Learning Tree Search approach for tactic prediction. By utilizing ATG4CI, we generate a LeanComb-Enhanced dataset comprising 260K combinatorial identities theorems, each with a complete formal proof in Lean, and experimental evaluations demonstrate that models trained on this dataset can generate more effective tactics, thereby improving success rates in automated theorem proving for combinatorial identities. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在正式定理证明方面取得了显著进展，但高质量训练数据的稀缺限制了其在复杂数学领域的能力。组合数学作为数学的基础，提供了分析离散结构和解决优化问题的重要工具。然而，其固有的复杂性使其特别适合组合恒等式的自动定理证明（ATP）具有挑战性。为解决这一问题，我们手动构建了LeanComb，这是一个用于组合恒等式的Lean基准测试集，据我们所知，这是首个专门构建用于组合恒等式的形式化定理证明基准测试集。我们开发了一种组合恒等式自动定理生成器ATG4CI，该生成器结合了一种自我提升的大规模语言模型建议的候选策略和基于强化学习树搜索的方法进行策略预测。利用ATG4CI，我们生成了包含26万条组合恒等式定理的LeanComb增强数据集，每条定理都有完整的Lean形式证明，并且实验评估表明，在此数据集上训练的模型可以生成更有效的策略，从而提高组合恒等式自动定理证明的成功率。 

---
# DocPuzzle: A Process-Aware Benchmark for Evaluating Realistic Long-Context Reasoning Capabilities 

**Title (ZH)**: DocPuzzle: 一种评估现实长上下文推理能力的过程感知基准 

**Authors**: Tianyi Zhuang, Chuqiao Kuang, Xiaoguang Li, Yihua Teng, Jihao Wu, Yasheng Wang, Lifeng Shang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17807)  

**Abstract**: We present DocPuzzle, a rigorously constructed benchmark for evaluating long-context reasoning capabilities in large language models (LLMs). This benchmark comprises 100 expert-level QA problems requiring multi-step reasoning over long real-world documents. To ensure the task quality and complexity, we implement a human-AI collaborative annotation-validation pipeline. DocPuzzle introduces an innovative evaluation framework that mitigates guessing bias through checklist-guided process analysis, establishing new standards for assessing reasoning capacities in LLMs. Our evaluation results show that: 1)Advanced slow-thinking reasoning models like o1-preview(69.7%) and DeepSeek-R1(66.3%) significantly outperform best general instruct models like Claude 3.5 Sonnet(57.7%); 2)Distilled reasoning models like DeepSeek-R1-Distill-Qwen-32B(41.3%) falls far behind the teacher model, suggesting challenges to maintain the generalization of reasoning capabilities relying solely on distillation. 

**Abstract (ZH)**: DocPuzzle: 一个严格构建的基准，用于评估大规模语言模型的长上下文推理能力 

---
# Detection of LLM-Paraphrased Code and Identification of the Responsible LLM Using Coding Style Features 

**Title (ZH)**: 基于编码风格特征的LLM重述代码检测及负责任的LLM识别 

**Authors**: Shinwoo Park, Hyundong Jin, Jeong-won Cha, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.17749)  

**Abstract**: Recent progress in large language models (LLMs) for code generation has raised serious concerns about intellectual property protection. Malicious users can exploit LLMs to produce paraphrased versions of proprietary code that closely resemble the original. While the potential for LLM-assisted code paraphrasing continues to grow, research on detecting it remains limited, underscoring an urgent need for detection system. We respond to this need by proposing two tasks. The first task is to detect whether code generated by an LLM is a paraphrased version of original human-written code. The second task is to identify which LLM is used to paraphrase the original code. For these tasks, we construct a dataset LPcode consisting of pairs of human-written code and LLM-paraphrased code using various LLMs.
We statistically confirm significant differences in the coding styles of human-written and LLM-paraphrased code, particularly in terms of naming consistency, code structure, and readability. Based on these findings, we develop LPcodedec, a detection method that identifies paraphrase relationships between human-written and LLM-generated code, and discover which LLM is used for the paraphrasing. LPcodedec outperforms the best baselines in two tasks, improving F1 scores by 2.64% and 15.17% while achieving speedups of 1,343x and 213x, respectively. 

**Abstract (ZH)**: 最近大语言模型在代码生成方面的进展引发了对知识产权保护的严重关切。恶意用户可以利用大语言模型生成与原版高度相似的变形代码。尽管大语言模型协助代码变形的潜力不断增长，但在这一检测方面的研究仍然有限，这突显了迫切需要开发检测系统。为此，我们提出了两个任务。第一个任务是检测由大语言模型生成的代码是否为人类编写代码的变形版本。第二个任务是识别使用了哪个大语言模型来变形原始代码。为这两个任务，我们构建了一个数据集LPcode，包含人类编写代码和由各种大语言模型生成的变形代码的配对。我们统计上证实了人类编写和大语言模型生成的代码在命名一致性、代码结构和可读性等方面的显著差异。基于这些发现，我们开发了LPcodedec检测方法，该方法可以识别人类编写和大语言模型生成代码之间的变形关系，并确定用于生成变形的哪个大语言模型。LPcodedec在两个任务上的表现优于最佳基线，分别提高了2.64%和15.17%的F1分数，同时分别实现了1343倍和213倍的速度提升。 

---
# Mind the Gesture: Evaluating AI Sensitivity to Culturally Offensive Non-Verbal Gestures 

**Title (ZH)**: 关注手势：评估AI对具有文化冒犯性的非言语手势的敏感性 

**Authors**: Akhila Yerukola, Saadia Gabriel, Nanyun Peng, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2502.17710)  

**Abstract**: Gestures are an integral part of non-verbal communication, with meanings that vary across cultures, and misinterpretations that can have serious social and diplomatic consequences. As AI systems become more integrated into global applications, ensuring they do not inadvertently perpetuate cultural offenses is critical. To this end, we introduce Multi-Cultural Set of Inappropriate Gestures and Nonverbal Signs (MC-SIGNS), a dataset of 288 gesture-country pairs annotated for offensiveness, cultural significance, and contextual factors across 25 gestures and 85 countries. Through systematic evaluation using MC-SIGNS, we uncover critical limitations: text-to-image (T2I) systems exhibit strong US-centric biases, performing better at detecting offensive gestures in US contexts than in non-US ones; large language models (LLMs) tend to over-flag gestures as offensive; and vision-language models (VLMs) default to US-based interpretations when responding to universal concepts like wishing someone luck, frequently suggesting culturally inappropriate gestures. These findings highlight the urgent need for culturally-aware AI safety mechanisms to ensure equitable global deployment of AI technologies. 

**Abstract (ZH)**: 多文化不合适手势和非言语信号数据集（MC-SIGNS）：揭示文化aware AI安全机制的迫切需求 

---
# From Perceptions to Decisions: Wildfire Evacuation Decision Prediction with Behavioral Theory-informed LLMs 

**Title (ZH)**: 从感知到决策：基于行为理论指导的大语言模型 wildfire 撤离决策预测 

**Authors**: Ruxiao Chen, Chenguang Wang, Yuran Sun, Xilei Zhao, Susu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17701)  

**Abstract**: Evacuation decision prediction is critical for efficient and effective wildfire response by helping emergency management anticipate traffic congestion and bottlenecks, allocate resources, and minimize negative impacts. Traditional statistical methods for evacuation decision prediction fail to capture the complex and diverse behavioral logic of different individuals. In this work, for the first time, we introduce FLARE, short for facilitating LLM for advanced reasoning on wildfire evacuation decision prediction, a Large Language Model (LLM)-based framework that integrates behavioral theories and models to streamline the Chain-of-Thought (CoT) reasoning and subsequently integrate with memory-based Reinforcement Learning (RL) module to provide accurate evacuation decision prediction and understanding. Our proposed method addresses the limitations of using existing LLMs for evacuation behavioral predictions, such as limited survey data, mismatching with behavioral theory, conflicting individual preferences, implicit and complex mental states, and intractable mental state-behavior mapping. Experiments on three post-wildfire survey datasets show an average of 20.47% performance improvement over traditional theory-informed behavioral models, with strong cross-event generalizability. Our complete code is publicly available at this https URL 

**Abstract (ZH)**: 高效的野生火灾疏散决策预测对于帮助紧急管理机构预见交通拥堵和瓶颈、合理分配资源并最小化负面影响至关重要。传统统计方法在预测疏散决策时无法捕捉不同个体复杂多样的行为逻辑。在此工作中，我们首次提出了FLARE，即促进大型语言模型进行野生火灾疏散决策预测的高级推理框架，该框架基于大型语言模型，整合行为理论和模型以简化Chain-of-Thought（CoT）推理，并结合基于记忆的强化学习模块，提供精确的疏散决策预测和理解。我们提出的方法解决了现有大型语言模型在疏散行为预测中的局限性，如有限的调查数据、理论匹配不一致、个体偏好冲突、复杂的心理状态以及难以解决的心理状态-行为映射问题。在三个野生火灾后的调查数据集上的实验显示，与传统的理论驱动行为模型相比，我们的方法平均提高了20.47%的性能，并具有较强的跨事件泛化能力。完整的代码已公开在该网址。 

---
# Socratic: Enhancing Human Teamwork via AI-enabled Coaching 

**Title (ZH)**: Socratic：通过AI赋能的教练技术提升人类团队协作能力 

**Authors**: Sangwon Seo, Bing Han, Rayan E. Harari, Roger D. Dias, Marco A. Zenati, Eduardo Salas, Vaibhav Unhelkar  

**Link**: [PDF](https://arxiv.org/pdf/2502.17643)  

**Abstract**: Coaches are vital for effective collaboration, but cost and resource constraints often limit their availability during real-world tasks. This limitation poses serious challenges in life-critical domains that rely on effective teamwork, such as healthcare and disaster response. To address this gap, we propose and realize an innovative application of AI: task-time team coaching. Specifically, we introduce Socratic, a novel AI system that complements human coaches by providing real-time guidance during task execution. Socratic monitors team behavior, detects misalignments in team members' shared understanding, and delivers automated interventions to improve team performance. We validated Socratic through two human subject experiments involving dyadic collaboration. The results demonstrate that the system significantly enhances team performance with minimal interventions. Participants also perceived Socratic as helpful and trustworthy, supporting its potential for adoption. Our findings also suggest promising directions both for AI research and its practical applications to enhance human teamwork. 

**Abstract (ZH)**: 教练对于有效的协作至关重要，但在现实任务中，成本和资源限制往往限制了其可用性。这种限制在依赖有效团队合作的生命攸关领域（如医疗保健和灾难响应）提出了严峻挑战。为解决这一问题，我们提出并实现了人工智能的一种创新应用：任务时间团队辅导。具体而言，我们引入了Socratic这一全新的人工智能系统，通过在任务执行过程中提供实时指导来补充人类教练。Socratic监控团队行为，检测团队成员之间共享理解的偏差，并通过自动化干预提高团队绩效。我们通过两项涉及双人协作的人类受控实验验证了Socratic的效果。结果表明，系统在尽量减少干预的情况下显著提升了团队绩效。参与者还认为Socratic非常有用和可信，支持其潜在的广泛应用。我们的研究结果还为人工智能研究及其增强人类团队合作的实际应用指出了积极的方向。 

---
# Representation Engineering for Large-Language Models: Survey and Research Challenges 

**Title (ZH)**: 大规模语言模型的表示工程：综述与研究挑战 

**Authors**: Lukasz Bartoszcze, Sarthak Munshi, Bryan Sukidi, Jennifer Yen, Zejia Yang, David Williams-King, Linh Le, Kosi Asuzu, Carsten Maple  

**Link**: [PDF](https://arxiv.org/pdf/2502.17601)  

**Abstract**: Large-language models are capable of completing a variety of tasks, but remain unpredictable and intractable. Representation engineering seeks to resolve this problem through a new approach utilizing samples of contrasting inputs to detect and edit high-level representations of concepts such as honesty, harmfulness or power-seeking. We formalize the goals and methods of representation engineering to present a cohesive picture of work in this emerging discipline. We compare it with alternative approaches, such as mechanistic interpretability, prompt-engineering and fine-tuning. We outline risks such as performance decrease, compute time increases and steerability issues. We present a clear agenda for future research to build predictable, dynamic, safe and personalizable LLMs. 

**Abstract (ZH)**: 大规模语言模型能够完成多种任务，但仍具有不可预测性和难以处理的特点。表示工程学通过利用对比输入样本的新方法来解决这一问题，旨在检测和编辑诸如诚实、危害性或权力追求等概念的高层表示。我们正式化表示工程学的目标和方法，呈现这一新兴学科工作的整体图景。我们将它与机械可解释性、提示工程和微调等替代方法进行比较。我们概述了可能出现的风险，如性能下降、计算时间增加和可控性问题。我们提出了一个明确的未来研究议程，旨在构建可预测、动态、安全和个人化的大规模语言模型。 

---
# Intention Recognition in Real-Time Interactive Navigation Maps 

**Title (ZH)**: 实时交互导航地图中的意图识别 

**Authors**: Peijie Zhao, Zunayed Arefin, Felipe Meneguzzi, Ramon Fraga Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2502.17581)  

**Abstract**: In this demonstration, we develop IntentRec4Maps, a system to recognise users' intentions in interactive maps for real-world navigation. IntentRec4Maps uses the Google Maps Platform as the real-world interactive map, and a very effective approach for recognising users' intentions in real-time. We showcase the recognition process of IntentRec4Maps using two different Path-Planners and a Large Language Model (LLM).
GitHub: this https URL 

**Abstract (ZH)**: 在本次演示中，我们开发了IntentRec4Maps系统，用于识别用户在实境导航互动地图中的意图。IntentRec4Maps使用Google Maps Platform作为实境互动地图，并采用一种非常有效的实时识别用户意图的方法。我们使用两种不同的路径规划器和一个大型语言模型（LLM）展示了IntentRec4Maps的识别过程。GitHub: this https URL。 

---
# How Do Large Language Monkeys Get Their Power (Laws)? 

**Title (ZH)**: 大型语言模型是如何获得其权力（法律）的？ 

**Authors**: Rylan Schaeffer, Joshua Kazdan, John Hughes, Jordan Juravsky, Sara Price, Aengus Lynch, Erik Jones, Robert Kirk, Azalia Mirhoseini, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2502.17578)  

**Abstract**: Recent research across mathematical problem solving, proof assistant programming and multimodal jailbreaking documents a striking finding: when (multimodal) language model tackle a suite of tasks with multiple attempts per task -- succeeding if any attempt is correct -- then the negative log of the average success rate scales a power law in the number of attempts. In this work, we identify an apparent puzzle: a simple mathematical calculation predicts that on each problem, the failure rate should fall exponentially with the number of attempts. We confirm this prediction empirically, raising a question: from where does aggregate polynomial scaling emerge? We then answer this question by demonstrating per-problem exponential scaling can be made consistent with aggregate polynomial scaling if the distribution of single-attempt success probabilities is heavy tailed such that a small fraction of tasks with extremely low success probabilities collectively warp the aggregate success trend into a power law - even as each problem scales exponentially on its own. We further demonstrate that this distributional perspective explains previously observed deviations from power law scaling, and provides a simple method for forecasting the power law exponent with an order of magnitude lower relative error, or equivalently, ${\sim}2-4$ orders of magnitude less inference compute. Overall, our work contributes to a better understanding of how neural language model performance improves with scaling inference compute and the development of scaling-predictable evaluations of (multimodal) language models. 

**Abstract (ZH)**: 近来关于数学问题求解、证明助手编程和多模态 Jailbreaking 的研究揭示了一个引人注目的发现：当（多模态）语言模型在每项任务中有多次尝试机会，并且只要有一次尝试正确即算成功时，平均成功率的负对数与尝试次数之间呈现出幂律关系。在本文中，我们识别出一个明显的悖论：一个简单的数学计算表明，每道题目的失败率应该随着尝试次数的增加而呈现出指数级下降。我们通过实验证实了这一预测，提出了一个疑问：整体幂律关系是如何出现的？我们通过证明如果单次尝试成功率的概率分布具有重尾特征，即使每道题目单独来看呈现出指数级下降的趋势，但一小部分具有极低成功率的任务可以集体影响整体成功率趋势，使其呈现出幂律关系，从而回答了这一问题。我们还证明了这种概率分布视角可以解释之前观察到的幂律关系偏离现象，并提供了一种使用相对误差低一个数量级的方法来预测幂律指数，或者等效地说，可减少约2到4个数量级的推理计算量。整体而言，我们的工作促进了对神经语言模型性能随推理计算量增加而提升机制的理解，并为（多模态）语言模型的可预测扩展性评估提供了贡献。 

---
# Dataset Featurization: Uncovering Natural Language Features through Unsupervised Data Reconstruction 

**Title (ZH)**: 数据集特征化：通过无监督数据重构发现自然语言特征 

**Authors**: Michal Bravansky, Vaclav Kubon, Suhas Hariharan, Robert Kirk  

**Link**: [PDF](https://arxiv.org/pdf/2502.17541)  

**Abstract**: Interpreting data is central to modern research. Large language models (LLMs) show promise in providing such natural language interpretations of data, yet simple feature extraction methods such as prompting often fail to produce accurate and versatile descriptions for diverse datasets and lack control over granularity and scale. To address these limitations, we propose a domain-agnostic method for dataset featurization that provides precise control over the number of features extracted while maintaining compact and descriptive representations comparable to human expert labeling. Our method optimizes the selection of informative binary features by evaluating the ability of an LLM to reconstruct the original data using those features. We demonstrate its effectiveness in dataset modeling tasks and through two case studies: (1) Constructing a feature representation of jailbreak tactics that compactly captures both the effectiveness and diversity of a larger set of human-crafted attacks; and (2) automating the discovery of features that align with human preferences, achieving accuracy and robustness comparable to expert-crafted features. Moreover, we show that the pipeline scales effectively, improving as additional features are sampled, making it suitable for large and diverse datasets. 

**Abstract (ZH)**: 现代研究中数据分析解释至关重要。大规模语言模型（LLMs）显示出了提供自然语言数据解释的潜力，然而简单的特征提取方法如提示常常无法为多样化的数据集生成准确且多样化的描述，并且缺乏对细节和规模的控制。为了解决这些局限性，我们提出了一种跨领域的数据集特征化方法，该方法能够精确控制提取的特征数量，同时保持紧凑且描述性的表示，与人类专家标注相媲美。我们的方法通过评估LLM使用这些特征重构原始数据的能力，来优化选择信息性二元特征。我们通过数据集建模任务和两个案例研究展示了其有效性：（1）构建一个包含更大规模人类设计攻击的有效性和多样性的狱破战术特征表示；（2）自动化发现与人类偏好相匹配的特征，实现与专家设计特征相当的准确性和鲁棒性。此外，我们展示了该流水线具有良好的扩展性，随着更多特征的采样而改进，使其适用于大规模和多样化的数据集。 

---
# User Intent to Use DeekSeep for Healthcare Purposes and their Trust in the Large Language Model: Multinational Survey Study 

**Title (ZH)**: 用户在健康护理目的下使用DeekSeep的意图及其对大型语言模型的信任：跨国调查研究 

**Authors**: Avishek Choudhury, Yeganeh Shahsavar, Hamid Shamszare  

**Link**: [PDF](https://arxiv.org/pdf/2502.17487)  

**Abstract**: Large language models (LLMs) increasingly serve as interactive healthcare resources, yet user acceptance remains underexplored. This study examines how ease of use, perceived usefulness, trust, and risk perception interact to shape intentions to adopt DeepSeek, an emerging LLM-based platform, for healthcare purposes. A cross-sectional survey of 556 participants from India, the United Kingdom, and the United States was conducted to measure perceptions and usage patterns. Structural equation modeling assessed both direct and indirect effects, including potential quadratic relationships. Results revealed that trust plays a pivotal mediating role: ease of use exerts a significant indirect effect on usage intentions through trust, while perceived usefulness contributes to both trust development and direct adoption. By contrast, risk perception negatively affects usage intent, emphasizing the importance of robust data governance and transparency. Notably, significant non-linear paths were observed for ease of use and risk, indicating threshold or plateau effects. The measurement model demonstrated strong reliability and validity, supported by high composite reliabilities, average variance extracted, and discriminant validity measures. These findings extend technology acceptance and health informatics research by illuminating the multifaceted nature of user adoption in sensitive domains. Stakeholders should invest in trust-building strategies, user-centric design, and risk mitigation measures to encourage sustained and safe uptake of LLMs in healthcare. Future work can employ longitudinal designs or examine culture-specific variables to further clarify how user perceptions evolve over time and across different regulatory environments. Such insights are critical for harnessing AI to enhance outcomes. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益成为互动型健康资源，但用户接受度尚未得到充分探索。本研究分析了便捷性、 perceived usefulness、信任和风险感知如何交互作用，进而影响用户采用基于LLM的新兴平台DeepSeek进行健康目的的意图。通过对来自印度、英国和美国的556名参与者的横断面调查，评估了感知和使用模式。结构方程建模评估了直接和间接影响，包括潜在的二次关系。研究结果揭示了信任起关键的中介作用：便捷性通过信任对使用意图产生显著的间接影响，而 perceived usefulness则促进了信任发展和直接采用。相比之下，风险感知消极地影响使用意图，强调了稳健的数据治理和透明度的重要性。值得注意的是，观察到便捷性和风险的显著非线性路径，表明存在阈值或平台效应。测量模型展示了强可靠性与有效性，这一点由较高的综合可靠性、抽取的平均方差和辨别性效度指标得到支持。这些发现扩展了技术接受和卫生信息学研究，揭示了敏感领域中用户采用的多维性质。利益相关者应投资于信任构建策略、以用户为中心的设计和风险缓解措施，以促进LLMs在医疗健康中的持续安全使用。未来的研究可以通过采用纵向设计或考察文化特定变量，进一步澄清用户感知如何随时间发展并在不同监管环境中变化。这些见解对于利用AI改善成果至关重要。 

---
# Scalable Equilibrium Sampling with Sequential Boltzmann Generators 

**Title (ZH)**: 可扩展的平衡采样方法：顺序玻尔兹曼生成器 

**Authors**: Charlie B. Tan, Avishek Joey Bose, Chen Lin, Leon Klein, Michael M. Bronstein, Alexander Tong  

**Link**: [PDF](https://arxiv.org/pdf/2502.18462)  

**Abstract**: Scalable sampling of molecular states in thermodynamic equilibrium is a long-standing challenge in statistical physics. Boltzmann generators tackle this problem by pairing powerful normalizing flows with importance sampling to obtain statistically independent samples under the target distribution. In this paper, we extend the Boltzmann generator framework and introduce Sequential Boltzmann generators (SBG) with two key improvements. The first is a highly efficient non-equivariant Transformer-based normalizing flow operating directly on all-atom Cartesian coordinates. In contrast to equivariant continuous flows of prior methods, we leverage exactly invertible non-equivariant architectures which are highly efficient both during sample generation and likelihood computation. As a result, this unlocks more sophisticated inference strategies beyond standard importance sampling. More precisely, as a second key improvement we perform inference-time scaling of flow samples using annealed Langevin dynamics which transports samples toward the target distribution leading to lower variance (annealed) importance weights which enable higher fidelity resampling with sequential Monte Carlo. SBG achieves state-of-the-art performance w.r.t. all metrics on molecular systems, demonstrating the first equilibrium sampling in Cartesian coordinates of tri, tetra, and hexapeptides that were so far intractable for prior Boltzmann generators. 

**Abstract (ZH)**: 在统计物理中，热力学平衡下的分子状态可扩展采样一直是长期挑战。波尔兹曼生成器通过将强大的归一化流动与重要性采样相结合，获得目标分布下的统计独立样本来应对这一问题。在本文中，我们扩展了波尔兹曼生成器框架，并引入了顺序波尔兹曼生成器（SBG），其中包括两个关键改进。首先，SBG采用高效的高度非等变Transformer基归一化流动，直接作用于全原子笛卡尔坐标。与先前方法中的等变连续流动相比，我们利用完全可逆的高度非等变架构，在样本生成和概率计算中均非常高效。这使得可以使用更复杂的推理策略，而不仅仅局限于标准的重要采样。其次，SBG在推理阶段利用退火拉梅尔动力学对流动样本进行缩放，将样本向目标分布转移，从而获得较低方差（退火）重要权重，使得顺序蒙特卡洛方法能够实现更高质量的重采样。SBG在分子系统上的所有指标上均实现了最佳性能，展示了首次在笛卡尔坐标中对三肽、四肽和六肽进行热力学平衡采样，这些肽对于先前的波尔兹曼生成器来说此前是无法处理的。 

---
# FRIDA to the Rescue! Analyzing Synthetic Data Effectiveness in Object-Based Common Sense Reasoning for Disaster Response 

**Title (ZH)**: FRIDA 挽救了！基于对象的常识推理中合成数据有效性分析在灾害响应中的应用 

**Authors**: Mollie Shichman, Claire Bonial, Austin Blodgett, Taylor Hudson, Francis Ferraro, Rachel Rudinger  

**Link**: [PDF](https://arxiv.org/pdf/2502.18452)  

**Abstract**: Large Language Models (LLMs) have the potential for substantial common sense reasoning. However, these capabilities are often emergent in larger models. This means smaller models that can be run locally are less helpful and capable with respect to certain reasoning tasks. To meet our problem space requirements, we fine-tune smaller LLMs to disaster domains, as these domains involve complex and low-frequency physical common sense knowledge. We introduce a pipeline to create Field Ready Instruction Decoding Agent (FRIDA) models, where domain experts and linguists combine their knowledge to make high-quality seed data that is used to generate synthetic data for fine-tuning. We create a set of 130 seed instructions for synthetic generation, a synthetic dataset of 25000 instructions, and 119 evaluation instructions relating to both general and earthquake-specific object affordances. We fine-tune several LLaMa and Mistral instruction-tuned models and find that FRIDA models outperform their base models at a variety of sizes. We then run an ablation study to understand which kinds of synthetic data most affect performance and find that training physical state and object function common sense knowledge alone improves over FRIDA models trained on all data. We conclude that the FRIDA pipeline is capable of instilling general common sense, but needs to be augmented with information retrieval for specific domain knowledge. 

**Abstract (ZH)**: 大型语言模型（LLMs）在常识推理方面具有潛力，但這些能力通常在較大的模型中 gradually 形成。这意味着较小的本地可运行模型在某些推理任务上较少有帮助。为了满足我们的问题空间要求，我们将较小的LLMs微调到灾害领域，因为这些领域涉及复杂的低频物理常识知识。我们提出了一种流水线来创建现场准备就绪指令解码代理（FRIDA）模型，其中领域专家和语言学家结合他们的知识生成高质量的种子数据，并用其生成合成数据以进行微调。我们创建了一组130个种子指令用于合成生成，一个包含25000个指令的合成数据集，以及涉及通用和地震特定物体功能性指令的119个评估指令。我们对LLaMa和Mistral指令微调模型进行了微调并发现，FRIDA模型在各种规模下均超越了其基础模型。然后我们进行了一项消融研究以了解哪种类型的合成数据对性能影响最大，发现单独训练物理状态和物体功能常识知识提高了FRIDA模型的性能。我们得出结论，FRIDA流水线可以灌输一般的常识，但需要增加信息检索来获取特定领域的知识。 

---
# SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution 

**Title (ZH)**: SWE-RL：通过开放软件演化中的强化学习提升大语言模型推理能力 

**Authors**: Yuxiang Wei, Olivier Duchenne, Jade Copet, Quentin Carbonneaux, Lingming Zhang, Daniel Fried, Gabriel Synnaeve, Rishabh Singh, Sida I. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18449)  

**Abstract**: The recent DeepSeek-R1 release has demonstrated the immense potential of reinforcement learning (RL) in enhancing the general reasoning capabilities of large language models (LLMs). While DeepSeek-R1 and other follow-up work primarily focus on applying RL to competitive coding and math problems, this paper introduces SWE-RL, the first approach to scale RL-based LLM reasoning for real-world software engineering. Leveraging a lightweight rule-based reward (e.g., the similarity score between ground-truth and LLM-generated solutions), SWE-RL enables LLMs to autonomously recover a developer's reasoning processes and solutions by learning from extensive open-source software evolution data -- the record of a software's entire lifecycle, including its code snapshots, code changes, and events such as issues and pull requests. Trained on top of Llama 3, our resulting reasoning model, Llama3-SWE-RL-70B, achieves a 41.0% solve rate on SWE-bench Verified -- a human-verified collection of real-world GitHub issues. To our knowledge, this is the best performance reported for medium-sized (<100B) LLMs to date, even comparable to leading proprietary LLMs like GPT-4o. Surprisingly, despite performing RL solely on software evolution data, Llama3-SWE-RL has even emerged with generalized reasoning skills. For example, it shows improved results on five out-of-domain tasks, namely, function coding, library use, code reasoning, mathematics, and general language understanding, whereas a supervised-finetuning baseline even leads to performance degradation on average. Overall, SWE-RL opens up a new direction to improve the reasoning capabilities of LLMs through reinforcement learning on massive software engineering data. 

**Abstract (ZH)**: Recent DeepSeek-R1版本展示了强化学习在提升大型语言模型通用推理能力方面的巨大潜力。DeepSeek-R1及其后续工作主要集中在将强化学习应用于编程和数学问题上，而本文介绍了SWE-RL，这是首个在实际软件工程中规模化应用基于强化学习的大型语言模型推理方法。通过利用轻量级规则为基础的奖励机制（如ground-truth与LLM生成解决方案的相似度得分），SWE-RL使LLMs能够通过学习开放源代码软件演化数据中的大量信息，自主恢复开发者的推理过程和解决方案——这些数据包括代码快照、代码变更以及诸如问题和拉取请求等事件。基于Llama 3训练，我们的推理模型Llama3-SWE-RL-70B在SWE-bench Verified（一个由人类验证的真实GitHub问题集合）上达到了41.0%的解题率。据我们所知，这在目前中型(<100B)LLM中是最佳性能，甚至可以与GPT-4o等领先私有LLM相媲美。令人惊讶的是，尽管仅在软件演化数据上进行RL训练，Llama3-SWE-RL仍展现出泛化的推理能力。例如，在五个离域任务（函数编码、库使用、代码推理、数学和通用语言理解）上表现出改进的结果，而监督微调基线在平均上甚至导致性能下降。总体而言，SWE-RL为通过大规模软件工程数据上的强化学习提高LLM推理能力开辟了新的方向。 

---
# Disambiguate First Parse Later: Generating Interpretations for Ambiguity Resolution in Semantic Parsing 

**Title (ZH)**: 先消歧后解析：语义解析中生成解释以解决歧义问题 

**Authors**: Irina Saparina, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2502.18448)  

**Abstract**: Handling ambiguity and underspecification is an important challenge in natural language interfaces, particularly for tasks like text-to-SQL semantic parsing. We propose a modular approach that resolves ambiguity using natural language interpretations before mapping these to logical forms (e.g., SQL queries). Although LLMs excel at parsing unambiguous utterances, they show strong biases for ambiguous ones, typically predicting only preferred interpretations. We constructively exploit this bias to generate an initial set of preferred disambiguations and then apply a specialized infilling model to identify and generate missing interpretations. To train the infilling model, we introduce an annotation method that uses SQL execution to validate different meanings. Our approach improves interpretation coverage and generalizes across datasets with different annotation styles, database structures, and ambiguity types. 

**Abstract (ZH)**: 处理自然语言接口中的模糊性和缺指性是自然语言处理任务的重要挑战，特别是在将文本转换为SQL语义解析的任务中。我们提出了一种模块化方法，首先使用自然语言解释来解决模糊性，然后再映射这些解释到逻辑形式（例如，SQL查询）。尽管大语言模型在解析无歧义的语句方面表现出色，但它们在歧义语句上的预测通常表现出强烈的偏好，通常只预测首选的解释。我们有建设性地利用这种偏好来生成一组首选的消歧解释，然后应用一个专门的填充模型来识别和生成缺失的解释。为了训练填充模型，我们引入了一种标注方法，该方法使用SQL执行来验证不同的意义。我们的方法提高了解释覆盖范围，并能在具有不同标注风格、数据库结构和模糊性类型的多个数据集中泛化。 

---
# ToMCAT: Theory-of-Mind for Cooperative Agents in Teams via Multiagent Diffusion Policies 

**Title (ZH)**: ToMCAT: Theory-of-Mind for Cooperative Agents in Teams via Multiagent Diffusion Policies-solving合作代理的理论心智通过多代理扩散策略的ToMCAT 

**Authors**: Pedro Sequeira, Vidyasagar Sadhu, Melinda Gervasio  

**Link**: [PDF](https://arxiv.org/pdf/2502.18438)  

**Abstract**: In this paper we present ToMCAT (Theory-of-Mind for Cooperative Agents in Teams), a new framework for generating ToM-conditioned trajectories. It combines a meta-learning mechanism, that performs ToM reasoning over teammates' underlying goals and future behavior, with a multiagent denoising-diffusion model, that generates plans for an agent and its teammates conditioned on both the agent's goals and its teammates' characteristics, as computed via ToM. We implemented an online planning system that dynamically samples new trajectories (replans) from the diffusion model whenever it detects a divergence between a previously generated plan and the current state of the world. We conducted several experiments using ToMCAT in a simulated cooking domain. Our results highlight the importance of the dynamic replanning mechanism in reducing the usage of resources without sacrificing team performance. We also show that recent observations about the world and teammates' behavior collected by an agent over the course of an episode combined with ToM inferences are crucial to generate team-aware plans for dynamic adaptation to teammates, especially when no prior information is provided about them. 

**Abstract (ZH)**: 基于理论共情的多智能体合作框架：ToMCAT 

---
# TextGames: Learning to Self-Play Text-Based Puzzle Games via Language Model Reasoning 

**Title (ZH)**: 文本游戏：通过语言模型推理学习自我对弈文本基础益智游戏 

**Authors**: Frederikus Hudi, Genta Indra Winata, Ruochen Zhang, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2502.18431)  

**Abstract**: Reasoning is a fundamental capability of large language models (LLMs), enabling them to comprehend, analyze, and solve complex problems. In this paper, we introduce TextGames, an innovative benchmark specifically crafted to assess LLMs through demanding text-based games that require advanced skills in pattern recognition, spatial awareness, arithmetic, and logical reasoning. Our analysis probes LLMs' performance in both single-turn and multi-turn reasoning, and their abilities in leveraging feedback to correct subsequent answers through self-reflection. Our findings reveal that, although LLMs exhibit proficiency in addressing most easy and medium-level problems, they face significant challenges with more difficult tasks. In contrast, humans are capable of solving all tasks when given sufficient time. Moreover, we observe that LLMs show improved performance in multi-turn predictions through self-reflection, yet they still struggle with sequencing, counting, and following complex rules consistently. Additionally, models optimized for reasoning outperform pre-trained LLMs that prioritize instruction following, highlighting the crucial role of reasoning skills in addressing highly complex problems. 

**Abstract (ZH)**: 大规模语言模型中推理能力的评估：TextGames基准研究 

---
# Comparative Analysis of MDL-VAE vs. Standard VAE on 202 Years of Gynecological Data 

**Title (ZH)**: MDL-VAE与标准VAE在202年妇科数据上的比较分析 

**Authors**: Paula Santos  

**Link**: [PDF](https://arxiv.org/pdf/2502.18412)  

**Abstract**: This study presents a comparative evaluation of a Variational Autoencoder (VAE) enhanced with Minimum Description Length (MDL) regularization against a Standard Autoencoder for reconstructing high-dimensional gynecological data. The MDL-VAE exhibits significantly lower reconstruction errors (MSE, MAE, RMSE) and more structured latent representations, driven by effective KL divergence regularization. Statistical analyses confirm these performance improvements are significant. Furthermore, the MDL-VAE shows consistent training and validation losses and achieves efficient inference times, underscoring its robustness and practical viability. Our findings suggest that incorporating MDL principles into VAE architectures can substantially improve data reconstruction and generalization, making it a promising approach for advanced applications in healthcare data modeling and analysis. 

**Abstract (ZH)**: 本研究对Minimum Description Length (MDL) 正则化增强的变分自编码器（MDL-VAE）与标准自编码器在重建高维妇科数据方面的性能进行了比较评估。统计分析证实了这些性能改进具有显著性。此外，MDL-VAE表现出一致的训练和验证损失，并实现了高效的推理时间，凸显了其鲁棒性和实际可行性。我们的研究结果表明，将MDL原理融入到变分自编码器架构中可以显著提高数据重建和泛化能力，使其成为高级健康数据分析应用的有前途的方法。 

---
# TSKANMixer: Kolmogorov-Arnold Networks with MLP-Mixer Model for Time Series Forecasting 

**Title (ZH)**: TSKANMixer: 拟 Kolmogorov-Arnold 网络与 MLP-Mixer 模型在时间序列预测中的应用 

**Authors**: Young-Chae Hong, Bei Xiao, Yangho Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18410)  

**Abstract**: Time series forecasting has long been a focus of research across diverse fields, including economics, energy, healthcare, and traffic management. Recent works have introduced innovative architectures for time series models, such as the Time-Series Mixer (TSMixer), which leverages multi-layer perceptrons (MLPs) to enhance prediction accuracy by effectively capturing both spatial and temporal dependencies within the data. In this paper, we investigate the capabilities of the Kolmogorov-Arnold Networks (KANs) for time-series forecasting by modifying TSMixer with a KAN layer (TSKANMixer). Experimental results demonstrate that TSKANMixer tends to improve prediction accuracy over the original TSMixer across multiple datasets, ranking among the top-performing models compared to other time series approaches. Our results show that the KANs are promising alternatives to improve the performance of time series forecasting by replacing or extending traditional MLPs. 

**Abstract (ZH)**: 时间序列预测一直是跨经济学、能源、医疗保健和交通管理等多元领域的研究重点。最近的研究引入了时间序列模型的新架构，如Time-Series Mixer (TSMixer)，其通过多层感知机（MLPs）有效捕捉数据中的时空依赖性以提高预测准确性。本文通过将Kolmogorov-Arnold Networks (KANs)层整合到TSMixer中，即TSKANMixer，来探究KANs在时间序列预测中的能力。实验结果表明，TSKANMixer在多个数据集上往往能提高预测准确性，并与其他时间序列方法相比名列前茅。我们的结果表明，KANs是改进时间序列预测性能的有前途的替代或扩展传统MLPs的选择。 

---
# AgentRM: Enhancing Agent Generalization with Reward Modeling 

**Title (ZH)**: AgentRM：通过奖励建模提升智能体泛化能力 

**Authors**: Yu Xia, Jingru Fan, Weize Chen, Siyu Yan, Xin Cong, Zhong Zhang, Yaxi Lu, Yankai Lin, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.18407)  

**Abstract**: Existing LLM-based agents have achieved strong performance on held-in tasks, but their generalizability to unseen tasks remains poor. Hence, some recent work focus on fine-tuning the policy model with more diverse tasks to improve the generalizability. In this work, we find that finetuning a reward model to guide the policy model is more robust than directly finetuning the policy model. Based on this finding, we propose AgentRM, a generalizable reward model, to guide the policy model for effective test-time search. We comprehensively investigate three approaches to construct the reward model, including explicit reward modeling, implicit reward modeling and LLM-as-a-judge. We then use AgentRM to guide the answer generation with Best-of-N sampling and step-level beam search. On four types of nine agent tasks, AgentRM enhances the base policy model by $8.8$ points on average, surpassing the top general agent by $4.0$. Moreover, it demonstrates weak-to-strong generalization, yielding greater improvement of $12.6$ on LLaMA-3-70B policy model. As for the specializability, AgentRM can also boost a finetuned policy model and outperform the top specialized agent by $11.4$ on three held-in tasks. Further analysis verifies its effectiveness in test-time scaling. Codes will be released to facilitate the research in this area. 

**Abstract (ZH)**: 基于LLM的奖励模型AgentRM在未见任务上的泛化和专用化能力研究 

---
# EgoSim: An Egocentric Multi-view Simulator and Real Dataset for Body-worn Cameras during Motion and Activity 

**Title (ZH)**: EgoSim：一种基于第一人称多视角的运动与活动穿戴相机模拟器及真实数据集 

**Authors**: Dominik Hollidt, Paul Streli, Jiaxi Jiang, Yasaman Haghighi, Changlin Qian, Xintong Liu, Christian Holz  

**Link**: [PDF](https://arxiv.org/pdf/2502.18373)  

**Abstract**: Research on egocentric tasks in computer vision has mostly focused on head-mounted cameras, such as fisheye cameras or embedded cameras inside immersive headsets. We argue that the increasing miniaturization of optical sensors will lead to the prolific integration of cameras into many more body-worn devices at various locations. This will bring fresh perspectives to established tasks in computer vision and benefit key areas such as human motion tracking, body pose estimation, or action recognition -- particularly for the lower body, which is typically occluded.
In this paper, we introduce EgoSim, a novel simulator of body-worn cameras that generates realistic egocentric renderings from multiple perspectives across a wearer's body. A key feature of EgoSim is its use of real motion capture data to render motion artifacts, which are especially noticeable with arm- or leg-worn cameras. In addition, we introduce MultiEgoView, a dataset of egocentric footage from six body-worn cameras and ground-truth full-body 3D poses during several activities: 119 hours of data are derived from AMASS motion sequences in four high-fidelity virtual environments, which we augment with 5 hours of real-world motion data from 13 participants using six GoPro cameras and 3D body pose references from an Xsens motion capture suit.
We demonstrate EgoSim's effectiveness by training an end-to-end video-only 3D pose estimation network. Analyzing its domain gap, we show that our dataset and simulator substantially aid training for inference on real-world data.
EgoSim code & MultiEgoView dataset: this https URL 

**Abstract (ZH)**: 基于计算机视觉的自我中心任务研究主要集中在头戴式摄像设备上，如鱼眼相机或内置在沉浸式头盔中的嵌入式相机。我们认为光学传感器的不断微型化将导致摄像头被集成到更多种类的身体佩戴设备中，并且分布于身体的不同部位。这将为计算机视觉中的传统任务带来新的视角，并且特别有利于人体动作追踪、身体姿态估计或动作识别——尤其是下肢，其通常被遮挡。  
在本文中，我们介绍了EgoSim，这是一种新颖的基于身体佩戴相机的模拟器，能够从穿戴者身体多个视角生成逼真的自我中心渲染。EgoSim的一个关键特征是使用真实的运动捕捉数据渲染运动伪像，这种伪像在佩戴于手臂或腿部的相机中尤为明显。此外，我们还引入了MultiEgoView数据集，该数据集包括六种身体佩戴相机的自我中心画面以及多种活动中的地面真值全身3D姿态：119小时的数据来源于在四个高保真虚拟环境中运动捕捉序列数据库AMASS中提取的数据，并通过13名参与者在六台GoPro相机下进行的真实世界运动数据以及Xsens动捕服提供的3D身体姿态参考进行扩充。  
我们通过训练一个端到端的仅视频输入的姿态估计网络来展示EgoSim的效果。通过分析其领域差距，我们证明我们的数据集和模拟器能够显著辅助现实世界数据上的推理训练。  
EgoSim代码及MultiEgoView数据集：https://doi.org/10.5281/zenodo.5653076 

---
# Which Contributions Deserve Credit? Perceptions of Attribution in Human-AI Co-Creation 

**Title (ZH)**: 哪些贡献值得 credited？人类与AI协同创作中的归因感知 

**Authors**: Jessica He, Stephanie Houde, Justin D. Weisz  

**Link**: [PDF](https://arxiv.org/pdf/2502.18357)  

**Abstract**: AI systems powered by large language models can act as capable assistants for writing and editing. In these tasks, the AI system acts as a co-creative partner, making novel contributions to an artifact-under-creation alongside its human partner(s). One question that arises in these scenarios is the extent to which AI should be credited for its contributions. We examined knowledge workers' views of attribution through a survey study (N=155) and found that they assigned different levels of credit across different contribution types, amounts, and initiative. Compared to a human partner, we observed a consistent pattern in which AI was assigned less credit for equivalent contributions. Participants felt that disclosing AI involvement was important and used a variety of criteria to make attribution judgments, including the quality of contributions, personal values, and technology considerations. Our results motivate and inform new approaches for crediting AI contributions to co-created work. 

**Abstract (ZH)**: 由大型语言模型驱动的AI系统可以作为写作和编辑的有能力的助手。在这些任务中，AI系统作为共创伙伴，在其人类同伴的陪伴下，对正在创造的作品做出新颖的贡献。这些场景中出现的一个问题是，AI应为其贡献获得多大程度的信用。我们通过调查研究（N=155）考察了知识工作者对归因的看法，并发现他们在不同类型的贡献、数量和主动性方面分配了不同的信用程度。与人类同伴相比，我们观察到一个一致的模式：AI在同等贡献中获得的信用较少。参与者认为披露AI的参与很重要，并使用多种标准来做出归因判断，包括贡献质量、个人价值观和技术考虑。我们的研究结果激励并指导了新的方法，用于承认共同创造工作中AI的贡献。 

---
# From Vision to Sound: Advancing Audio Anomaly Detection with Vision-Based Algorithms 

**Title (ZH)**: 从视觉到声音：基于视觉算法推进音频异常检测 

**Authors**: Manuel Barusco, Francesco Borsatti, Davide Dalle Pezze, Francesco Paissan, Elisabetta Farella, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2502.18328)  

**Abstract**: Recent advances in Visual Anomaly Detection (VAD) have introduced sophisticated algorithms leveraging embeddings generated by pre-trained feature extractors. Inspired by these developments, we investigate the adaptation of such algorithms to the audio domain to address the problem of Audio Anomaly Detection (AAD). Unlike most existing AAD methods, which primarily classify anomalous samples, our approach introduces fine-grained temporal-frequency localization of anomalies within the spectrogram, significantly improving explainability. This capability enables a more precise understanding of where and when anomalies occur, making the results more actionable for end users. We evaluate our approach on industrial and environmental benchmarks, demonstrating the effectiveness of VAD techniques in detecting anomalies in audio signals. Moreover, they improve explainability by enabling localized anomaly identification, making audio anomaly detection systems more interpretable and practical. 

**Abstract (ZH)**: 近期在视觉异常检测(VAD)领域的进展引入了利用预训练特征提取器生成的嵌入的复杂算法。受此启发，我们探讨将此类算法适应到音频领域以解决音频异常检测(AAD)问题。与大多数现有AAD方法主要对异常样本进行分类不同，我们的方法引入了谱图中异常的细粒度时间-频率定位，显著提高了可解释性。这种能力使得对异常发生的时间和位置有更精确的理解，使结果更可行地应用于最终用户。我们在工业和环境基准上评估了我们的方法，证明了VAD技术在检测音频信号异常方面的有效性。此外，它们通过实现局部异常识别提高了可解释性，使音频异常检测系统更具可解释性和实用性。 

---
# Smart and Efficient IoT-Based Irrigation System Design: Utilizing a Hybrid Agent-Based and System Dynamics Approach 

**Title (ZH)**: 基于混合基于代理和系统动力学方法的智能高效物联网灌溉系统设计 

**Authors**: Taha Ahmadi Pargo, Mohsen Akbarpour Shirazi, Dawud Fadai  

**Link**: [PDF](https://arxiv.org/pdf/2502.18298)  

**Abstract**: Regarding problems like reduced precipitation and an increase in population, water resource scarcity has become one of the most critical problems in modern-day societies, as a consequence, there is a shortage of available water resources for irrigation in arid and semi-arid countries. On the other hand, it is possible to utilize modern technologies to control irrigation and reduce water loss. One of these technologies is the Internet of Things (IoT). Despite the possibility of using the IoT in irrigation control systems, there are complexities in designing such systems. Considering this issue, it is possible to use agent-oriented software engineering (AOSE) methodologies to design complex cyber-physical systems such as IoT-based systems. In this research, a smart irrigation system is designed based on Prometheus AOSE methodology, to reduce water loss by maintaining soil moisture in a suitable interval. The designed system comprises sensors, a central agent, and irrigation nodes. These agents follow defined rules to maintain soil moisture at a desired level cooperatively. For system simulation, a hybrid agent-based and system dynamics model was designed. In this hybrid model, soil moisture dynamics were modeled based on the system dynamics approach. The proposed model, was implemented in AnyLogic computer simulation software. Utilizing the simulation model, irrigation rules were examined. The system's functionality in automatic irrigation mode was tested based on a 256-run, fractional factorial design, and the effects of important factors such as soil properties on total irrigated water and total operation time were analyzed. Based on the tests, the system consistently irrigated nearly optimal water amounts in all tests. Moreover, the results were also used to minimize the system's energy consumption by reducing the system's operational time. 

**Abstract (ZH)**: 基于普罗米修斯本体导向软件工程方法的智能灌溉系统设计与分析 

---
# Mixing Any Cocktail with Limited Ingredients: On the Structure of Payoff Sets in Multi-Objective MDPs and its Impact on Randomised Strategies 

**Title (ZH)**: 用有限的原料调出任何鸡尾酒：多目标MDPs中收益集的结构及其对随机化策略的影响 

**Authors**: James C. A. Main, Mickael Randour  

**Link**: [PDF](https://arxiv.org/pdf/2502.18296)  

**Abstract**: We consider multi-dimensional payoff functions in Markov decision processes, and ask whether a given expected payoff vector can be achieved or not. In general, pure strategies (i.e., not resorting to randomisation) do not suffice for this problem.
We study the structure of the set of expected payoff vectors of all strategies given a multi-dimensional payoff function and its consequences regarding randomisation requirements for strategies. In particular, we prove that for any payoff for which the expectation is well-defined under all strategies, it is sufficient to mix (i.e., randomly select a pure strategy at the start of a play and committing to it for the rest of the play) finitely many pure strategies to approximate any expected payoff vector up to any precision. Furthermore, for any payoff for which the expected payoff is finite under all strategies, any expected payoff can be obtained exactly by mixing finitely many strategies. 

**Abstract (ZH)**: 我们在马尔可夫决策过程中的多维收益函数下考虑给定的期望收益向量能否被实现的问题，并证明纯策略（即不采用随机化）通常不足以解决问题。我们研究给定多维收益函数下的所有策略的期望收益向量集的结构及其对策略中随机化要求的影响。特别地，我们证明对于任何收益，在所有策略下其期望值都是有定义的，只需混合有限个纯策略即可在任意精度上近似任何期望收益向量。进一步地，对于任何在所有策略下期望收益都有限的收益，只需混合有限个策略即可精确获得任何期望收益。 

---
# AMPO: Active Multi-Preference Optimization 

**Title (ZH)**: AMPO: 主动多偏好优化 

**Authors**: Taneesh Gupta, Rahul Madhavan, Xuchao Zhang, Chetan Bansal, Saravan Rajmohan  

**Link**: [PDF](https://arxiv.org/pdf/2502.18293)  

**Abstract**: Multi-preference optimization enriches language-model alignment beyond pairwise preferences by contrasting entire sets of helpful and undesired responses, thereby enabling richer training signals for large language models. During self-play alignment, these models often produce numerous candidate answers per query, rendering it computationally infeasible to include all responses in the training objective. In this work, we propose $\textit{Active Multi-Preference Optimization}$ (AMPO), a novel approach that combines on-policy generation, a multi-preference group-contrastive loss, and active subset selection. Specifically, we score and embed large candidate pools of responses and then select a small, yet informative, subset that covers reward extremes and distinct semantic clusters for preference optimization. Our contrastive training scheme is capable of identifying not only the best and worst answers but also subtle, underexplored modes that are crucial for robust alignment. Theoretically, we provide guarantees for expected reward maximization using our active selection method, and empirically, AMPO achieves state-of-the-art results on $\textit{AlpacaEval}$ using Llama 8B. 

**Abstract (ZH)**: 多偏好优化扩展了大型语言模型对齐的方法，通过对比整个有益和不期望回答的集合，超越了成对偏好，从而为大型语言模型提供更丰富的训练信号。在自我对齐过程中，这些模型经常为每个查询生成众多候选答案，使包括所有回答在训练目标中变得计算上不可行。在本文中，我们提出了一种名为$\textit{Active Multi-Preference Optimization}$（AMPO）的新方法，该方法结合了策略生成、多偏好群体对比损失以及主动子集选择。具体地，我们对大型候选回答池进行评分和嵌入，然后选择一个小而具有信息量的子集，该子集涵盖了奖励极值和不同的语义簇，用于偏好优化。我们的对比训练方案不仅能够识别最佳和最差的答案，还能识别对于稳健对齐至关重要的未开发模式。从理论上，我们提供了使用我们主动选择方法进行预期奖励最大化的保证；从实验上，AMPO在使用Llama 8B模型进行$\textit{AlpacaEval}$测试时达到了最先进的结果。 

---
# A Reverse Mamba Attention Network for Pathological Liver Segmentation 

**Title (ZH)**: 反转乌眼镜蛇注意力网络在病理性肝脏分割中的应用 

**Authors**: Jun Zeng, Ulas Bagci, Debesh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2502.18232)  

**Abstract**: We present RMA-Mamba, a novel architecture that advances the capabilities of vision state space models through a specialized reverse mamba attention module (RMA). The key innovation lies in RMA-Mamba's ability to capture long-range dependencies while maintaining precise local feature representation through its hierarchical processing pipeline. By integrating Vision Mamba (VMamba)'s efficient sequence modeling with RMA's targeted feature refinement, our architecture achieves superior feature learning across multiple scales. This dual-mechanism approach enables robust handling of complex morphological patterns while maintaining computational efficiency. We demonstrate RMA-Mamba's effectiveness in the challenging domain of pathological liver segmentation (from both CT and MRI), where traditional segmentation approaches often fail due to tissue variations. When evaluated on a newly introduced cirrhotic liver dataset (CirrMRI600+) of T2-weighted MRI scans, RMA-Mamba achieves the state-of-the-art performance with a Dice coefficient of 92.08%, mean IoU of 87.36%, and recall of 92.96%. The architecture's generalizability is further validated on the cancerous liver segmentation from CT scans (LiTS: Liver Tumor Segmentation dataset), yielding a Dice score of 92.9% and mIoU of 88.99%. The source code of the proposed RMA-Mamba is available at this https URL. 

**Abstract (ZH)**: RMA-Mamba：一种通过专门的反蟒蛇注意力模块（RMA）推进视觉状态空间模型能力的新型架构 

---
# Liver Cirrhosis Stage Estimation from MRI with Deep Learning 

**Title (ZH)**: 基于深度学习的肝脏 cirrhosis 阶段从 MRI 估计 

**Authors**: Jun Zeng, Debesh Jha, Ertugrul Aktas, Elif Keles, Alpay Medetalibeyoglu, Matthew Antalek, Amir A. Borhani, Daniela P. Ladner, Gorkem Durak, Ulas Bagci  

**Link**: [PDF](https://arxiv.org/pdf/2502.18225)  

**Abstract**: We present an end-to-end deep learning framework for automated liver cirrhosis stage estimation from multi-sequence MRI. Cirrhosis is the severe scarring (fibrosis) of the liver and a common endpoint of various chronic liver diseases. Early diagnosis is vital to prevent complications such as decompensation and cancer, which significantly decreases life expectancy. However, diagnosing cirrhosis in its early stages is challenging, and patients often present with life-threatening complications. Our approach integrates multi-scale feature learning with sequence-specific attention mechanisms to capture subtle tissue variations across cirrhosis progression stages. Using CirrMRI600+, a large-scale publicly available dataset of 628 high-resolution MRI scans from 339 patients, we demonstrate state-of-the-art performance in three-stage cirrhosis classification. Our best model achieves 72.8% accuracy on T1W and 63.8% on T2W sequences, significantly outperforming traditional radiomics-based approaches. Through extensive ablation studies, we show that our architecture effectively learns stage-specific imaging biomarkers. We establish new benchmarks for automated cirrhosis staging and provide insights for developing clinically applicable deep learning systems. The source code will be available at this https URL. 

**Abstract (ZH)**: 一种用于多序列MRI自动化肝硬化阶段估计的端到端深度学习框架 

---
# UASTrack: A Unified Adaptive Selection Framework with Modality-Customization in Single Object Tracking 

**Title (ZH)**: UASTrack：具备模态自适应定制的统一单目标跟踪框架 

**Authors**: He Wang, Tianyang Xu, Zhangyong Tang, Xiao-Jun Wu, Josef Kittler  

**Link**: [PDF](https://arxiv.org/pdf/2502.18220)  

**Abstract**: Multi-modal tracking is essential in single-object tracking (SOT), as different sensor types contribute unique capabilities to overcome challenges caused by variations in object appearance. However, existing unified RGB-X trackers (X represents depth, event, or thermal modality) either rely on the task-specific training strategy for individual RGB-X image pairs or fail to address the critical importance of modality-adaptive perception in real-world applications. In this work, we propose UASTrack, a unified adaptive selection framework that facilitates both model and parameter unification, as well as adaptive modality discrimination across various multi-modal tracking tasks. To achieve modality-adaptive perception in joint RGB-X pairs, we design a Discriminative Auto-Selector (DAS) capable of identifying modality labels, thereby distinguishing the data distributions of auxiliary modalities. Furthermore, we propose a Task-Customized Optimization Adapter (TCOA) tailored to various modalities in the latent space. This strategy effectively filters noise redundancy and mitigates background interference based on the specific characteristics of each modality. Extensive comparisons conducted on five benchmarks including LasHeR, GTOT, RGBT234, VisEvent, and DepthTrack, covering RGB-T, RGB-E, and RGB-D tracking scenarios, demonstrate our innovative approach achieves comparative performance by introducing only additional training parameters of 1.87M and flops of 1.95G. The code will be available at this https URL. 

**Abstract (ZH)**: 多模态跟踪是单对象跟踪（SOT）中的关键要素，不同传感器类型能提供独特的功能，以克服由于对象外观变化引起的各种挑战。然而，现有的统一RGB-X跟踪器（X代表深度、事件或热成像模态）要么依赖于针对特定任务的训练策略仅用于单个RGB-X图像对，要么未能解决在真实世界应用中多模态适应感知的关键重要性。在本文中，我们提出UASTrack，这是一种统一的自适应选择框架，能够促进模型和参数的统一，以及在各种多模态跟踪任务中实现自适应模态区分。为实现联合RGB-X图像对中的模态适应感知，我们设计了一种鉴别自选择器（DAS），能够识别模态标签，从而区分辅助模态的数据分布。此外，我们提出了一个针对不同模态定制的优化适配器（TCOA），在潜空间中适应各种模态。该策略根据每种模态的特定特征有效过滤噪声冗余并减轻背景干扰。在LasHeR、GTOT、RGBT234、VisEvent和DepthTrack五个基准数据集中进行的广泛比较，涵盖了RGB-T、RGB-E和RGB-D跟踪场景，证明我们的创新方法仅通过引入1.87M的额外训练参数和1.95G的FLOPS，就能达到相当的性能。代码将在此处提供：这个链接。 

---
# FLARE: A Framework for Stellar Flare Forecasting using Stellar Physical Properties and Historical Records 

**Title (ZH)**: FLARE：基于恒星物理性质和历史记录的耀斑预测框架 

**Authors**: Bingke Zhu, Xiaoxiao Wang, Minghui Jia, Yihan Tao, Xiao Kong, Ali Luo, Yingying Chen, Ming Tang, Jinqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18218)  

**Abstract**: Stellar flare events are critical observational samples for astronomical research; however, recorded flare events remain limited. Stellar flare forecasting can provide additional flare event samples to support research efforts. Despite this potential, no specialized models for stellar flare forecasting have been proposed to date. In this paper, we present extensive experimental evidence demonstrating that both stellar physical properties and historical flare records are valuable inputs for flare forecasting tasks. We then introduce FLARE (Forecasting Light-curve-based Astronomical Records via features Ensemble), the first-of-its-kind large model specifically designed for stellar flare forecasting. FLARE integrates stellar physical properties and historical flare records through a novel Soft Prompt Module and Residual Record Fusion Module. Our experiments on the publicly available Kepler light curve dataset demonstrate that FLARE achieves superior performance compared to other methods across all evaluation metrics. Finally, we validate the forecast capability of our model through a comprehensive case study. 

**Abstract (ZH)**: 恒星耀斑事件是天文学研究中重要的观测样本；然而，记录的耀斑事件仍有限。恒星耀斑预测可以提供额外的耀斑事件样本以支持研究工作。尽管存在这种潜力，但目前尚未提出专门用于恒星耀斑预测的模型。本文提供了广泛实验证据，证明恒星物理性质和历史耀斑记录都是耀斑预测任务的重要输入。我们随后介绍了FLARE（通过特征集成预测基于光曲线的天文记录），这是首个专门设计用于恒星耀斑预测的大型模型。FLARE通过新颖的Soft Prompt Module和残差记录融合模块整合恒星物理性质和历史耀斑记录。我们在公开可用的Kepler光曲线数据集上的实验表明，FLARE在所有评估指标中均表现出优越性能。最后，我们通过全面的案例研究验证了我们模型的预测能力。 

---
# LAG: LLM agents for Leaderboard Auto Generation on Demanding 

**Title (ZH)**: LAG: LLM代理在具有挑战性的排行榜自动生成中 

**Authors**: Jian Wu, Jiayu Zhang, Dongyuan Li, Linyi Yang, Aoxiao Zhong, Renhe Jiang, Qingsong Wen, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18209)  

**Abstract**: This paper introduces Leaderboard Auto Generation (LAG), a novel and well-organized framework for automatic generation of leaderboards on a given research topic in rapidly evolving fields like Artificial Intelligence (AI). Faced with a large number of AI papers updated daily, it becomes difficult for researchers to track every paper's proposed methods, experimental results, and settings, prompting the need for efficient automatic leaderboard construction. While large language models (LLMs) offer promise in automating this process, challenges such as multi-document summarization, leaderboard generation, and experiment fair comparison still remain under exploration. LAG solves these challenges through a systematic approach that involves the paper collection, experiment results extraction and integration, leaderboard generation, and quality evaluation. Our contributions include a comprehensive solution to the leaderboard construction problem, a reliable evaluation method, and experimental results showing the high quality of leaderboards. 

**Abstract (ZH)**: Leaderboard 自动生成（LAG）：一种适用于快速发展的人工智能研究领域的新型有效框架 

---
# DenoMAE2.0: Improving Denoising Masked Autoencoders by Classifying Local Patches 

**Title (ZH)**: DenoMAE2.0: 通过分类局部 patches 提高去噪掩蔽自编码器性能 

**Authors**: Atik Faysal, Mohammad Rostami, Taha Boushine, Reihaneh Gh. Roshan, Huaxia Wang, Nikhil Muralidhar  

**Link**: [PDF](https://arxiv.org/pdf/2502.18202)  

**Abstract**: We introduce DenoMAE2.0, an enhanced denoising masked autoencoder that integrates a local patch classification objective alongside traditional reconstruction loss to improve representation learning and robustness. Unlike conventional Masked Autoencoders (MAE), which focus solely on reconstructing missing inputs, DenoMAE2.0 introduces position-aware classification of unmasked patches, enabling the model to capture fine-grained local features while maintaining global coherence. This dual-objective approach is particularly beneficial in semi-supervised learning for wireless communication, where high noise levels and data scarcity pose significant challenges. We conduct extensive experiments on modulation signal classification across a wide range of signal-to-noise ratios (SNRs), from extremely low to moderately high conditions and in a low data regime. Our results demonstrate that DenoMAE2.0 surpasses its predecessor, Deno-MAE, and other baselines in both denoising quality and downstream classification accuracy. DenoMAE2.0 achieves a 1.1% improvement over DenoMAE on our dataset and 11.83%, 16.55% significant improved accuracy gains on the RadioML benchmark, over DenoMAE, for constellation diagram classification of modulation signals. 

**Abstract (ZH)**: DenoMAE2.0: 增强的去噪掩蔽自编码器及其在无线通信半监督学习中的应用 

---
# VesselSAM: Leveraging SAM for Aortic Vessel Segmentation with LoRA and Atrous Attention 

**Title (ZH)**: VesselSAM: 利用SAM进行主动脉血管分割的LoRA和空洞注意机制 

**Authors**: Adnan Iltaf, Rayan Merghani Ahmed, Bin Li, Shoujun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.18185)  

**Abstract**: Medical image segmentation is crucial for clinical diagnosis and treatment planning, particularly for complex anatomical structures like vessels. In this work, we propose VesselSAM, a modified version of the Segmentation Anything Model (SAM), specifically designed for aortic vessel segmentation. VesselSAM incorporates AtrousLoRA, a novel module that combines Atrous Attention with Low-Rank Adaptation (LoRA), to improve segmentation performance. Atrous Attention enables the model to capture multi-scale contextual information, preserving both fine local details and broader global context. At the same time, LoRA facilitates efficient fine-tuning of the frozen SAM image encoder, reducing the number of trainable parameters and ensuring computational efficiency. We evaluate VesselSAM on two challenging datasets: the Aortic Vessel Tree (AVT) dataset and the Type-B Aortic Dissection (TBAD) dataset. VesselSAM achieves state-of-the-art performance with DSC scores of 93.50\%, 93.25\%, 93.02\%, and 93.26\% across multiple medical centers. Our results demonstrate that VesselSAM delivers high segmentation accuracy while significantly reducing computational overhead compared to existing large-scale models. This development paves the way for enhanced AI-based aortic vessel segmentation in clinical environments. The code and models will be released at this https URL. 

**Abstract (ZH)**: VesselSAM：一种用于主动脉血管分割的改良Segmentation Anything Model 

---
# Problem Solved? Information Extraction Design Space for Layout-Rich Documents using LLMs 

**Title (ZH)**: 问题解决了吗？使用LLMs的布局丰富文档的信息提取设计空间 

**Authors**: Gaye Colakoglu, Gürkan Solmaz, Jonathan Fürst  

**Link**: [PDF](https://arxiv.org/pdf/2502.18179)  

**Abstract**: This paper defines and explores the design space for information extraction (IE) from layout-rich documents using large language models (LLMs). The three core challenges of layout-aware IE with LLMs are 1) data structuring, 2) model engagement, and 3) output refinement. Our study delves into the sub-problems within these core challenges, such as input representation, chunking, prompting, and selection of LLMs and multimodal models. It examines the outcomes of different design choices through a new layout-aware IE test suite, benchmarking against the state-of-art (SoA) model LayoutLMv3. The results show that the configuration from one-factor-at-a-time (OFAT) trial achieves near-optimal results with 14.1 points F1-score gain from the baseline model, while full factorial exploration yields only a slightly higher 15.1 points gain at around 36x greater token usage. We demonstrate that well-configured general-purpose LLMs can match the performance of specialized models, providing a cost-effective alternative. Our test-suite is freely available at this https URL. 

**Abstract (ZH)**: 本文定义并探索了使用大型语言模型（LLMs）从布局丰富文档中提取信息（IE）的设计空间。布局感知IE的核心挑战包括1）数据结构化、2）模型参与，以及3）输出精炼。本研究深入探讨了这些核心挑战下的子问题，如输入表示、切分、提示以及LLMs和多模态模型的选择。通过一个新的布局感知IE测试套件，本研究基于最新的基准模型LayoutLMv3进行评估，考察了不同设计选择的结果。结果显示，单一因素试验（OFAT）配置从基线模型获得了14.1点的F1分数提升，而全面的因素探索则在大约36倍的令牌使用量下只获得了略高的15.1点提升。本研究证明，配置良好的通用语言模型可以匹配专业模型的性能，提供一种成本效益较高的替代方案。测试套件在此网址免费获取。 

---
# CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification 

**Title (ZH)**: CLIPure：通过CLIP在潜在空间中净化实现对抗稳健的零样本分类 

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.18176)  

**Abstract**: In this paper, we aim to build an adversarially robust zero-shot image classifier. We ground our work on CLIP, a vision-language pre-trained encoder model that can perform zero-shot classification by matching an image with text prompts ``a photo of a <class-name>.''. Purification is the path we choose since it does not require adversarial training on specific attack types and thus can cope with any foreseen attacks. We then formulate purification risk as the KL divergence between the joint distributions of the purification process of denoising the adversarial samples and the attack process of adding perturbations to benign samples, through bidirectional Stochastic Differential Equations (SDEs). The final derived results inspire us to explore purification in the multi-modal latent space of CLIP. We propose two variants for our CLIPure approach: CLIPure-Diff which models the likelihood of images' latent vectors with the DiffusionPrior module in DaLLE-2 (modeling the generation process of CLIP's latent vectors), and CLIPure-Cos which models the likelihood with the cosine similarity between the embeddings of an image and ``a photo of a.''. As far as we know, CLIPure is the first purification method in multi-modal latent space and CLIPure-Cos is the first purification method that is not based on generative models, which substantially improves defense efficiency. We conducted extensive experiments on CIFAR-10, ImageNet, and 13 datasets that previous CLIP-based defense methods used for evaluating zero-shot classification robustness. Results show that CLIPure boosts the SOTA robustness by a large margin, e.g., from 71.7% to 91.1% on CIFAR10, from 59.6% to 72.6% on ImageNet, and 108% relative improvements of average robustness on the 13 datasets over previous SOTA. The code is available at this https URL. 

**Abstract (ZH)**: 基于CLIP的多模态-latent空间对抗鲁棒零样本图像分类方法CLIPure 

---
# SECURA: Sigmoid-Enhanced CUR Decomposition with Uninterrupted Retention and Low-Rank Adaptation in Large Language Models 

**Title (ZH)**: SECURA：增强的Sigmoid-CUR分解方法，保留不间断的低秩适应性在大规模语言模型中 

**Authors**: Zhang Yuxuan, Li Ruizhe  

**Link**: [PDF](https://arxiv.org/pdf/2502.18168)  

**Abstract**: With the rapid development of large language models (LLMs), fully fine-tuning (FT) these models has become increasingly impractical due to the high computational demands. Additionally, FT can lead to catastrophic forgetting. As an alternative, Low-Rank Adaptation (LoRA) has been proposed, which fine-tunes only a small subset of parameters, achieving similar performance to FT while significantly reducing resource requirements. However, since LoRA inherits FT's design, the issue of catastrophic forgetting remains.
To address these challenges, we propose SECURA: Sigmoid-Enhanced CUR Decomposition LoRA, a novel parameter-efficient fine-tuning (PEFT) variant that mitigates catastrophic forgetting while improving fine-tuning performance. Our method introduces a new normalization technique, SigNorm, to enhance parameter retention and overall performance.
SECURA has been evaluated on a variety of tasks, including mathematical problem-solving (GSM8K), challenging question-answering (CNNDM), translation (NewsDE), and complex multiple-choice reasoning (LogiQA). Experimental results show that SECURA achieves an average fine-tuning improvement of 3.59% across four multiple-choice question (MCQ) tasks and a 2.51% improvement across five question-answering (QA) tasks on models such as Gemma2 2b, Qwen2 1.5b, Qwen 2 7b, Llama3 8b, and Llama3.1 8b, compared to DoRA. Moreover, SECURA demonstrates superior knowledge retention capabilities, maintaining more than 70% accuracy on basic LLM knowledge across 16 continual learning tests, outperforming Experience Replay (ER), Sequential Learning (SEQ), EWC, I-LoRA, and CUR-LoRA. 

**Abstract (ZH)**: 随着大型语言模型（LLM）的迅速发展，全量微调（FT）这些模型因高计算需求而变得越来越不现实，同时FT还会导致灾难性遗忘。作为替代方案，低秩适应（LoRA）被提出，它仅微调一小部分参数，从而在显著减少资源需求的同时达到与FT相似的性能。然而，由于LoRA继承了FT的设计，灾难性遗忘的问题仍然存在。

为解决这些挑战，我们提出了SECURA：Sigmoid-Enhanced CUR分解LoRA，一种新的参数高效微调（PEFT）变体，能够在缓解灾难性遗忘的同时提升微调性能。该方法引入了一种新的归一化技术SigNorm，以增强参数保留和总体性能。

实验结果表明，SECURA在多种任务上均表现出色，包括数学问题解决（GSM8K）、挑战性问答（CNNDM）、翻译（NewsDE）和复杂多项选择推理（LogiQA）。实验结果显示，在Gemma2 2b、Qwen2 1.5b、Qwen 2 7b、Llama3 8b和Llama3.1 8b等模型上，与DoRA相比，SECURA在四个多项选择问题（MCQ）任务上的微调性能平均提高了3.59%，在五个问答（QA）任务上的性能提高了2.51%。此外，SECURA在基本LLM知识上的知识保留能力更优，在16次连续学习测试中保持了超过70%的准确性，优于Experience Replay（ER）、Sequential Learning（SEQ）、EWC、I-LoRA和CUR-LoRA。 

---
# iTrash: Incentivized Token Rewards for Automated Sorting and Handling 

**Title (ZH)**: iTrash: 基于激励的代币奖励自动分类与处理 

**Authors**: Pablo Ortega, Eduardo Castelló Ferrer  

**Link**: [PDF](https://arxiv.org/pdf/2502.18161)  

**Abstract**: As robotic systems (RS) become more autonomous, they are becoming increasingly used in small spaces and offices to automate tasks such as cleaning, infrastructure maintenance, or resource management. In this paper, we propose iTrash, an intelligent trashcan that aims to improve recycling rates in small office spaces. For that, we ran a 5 day experiment and found that iTrash can produce an efficiency increase of more than 30% compared to traditional trashcans. The findings derived from this work, point to the fact that using iTrash not only increase recyclying rates, but also provides valuable data such as users behaviour or bin usage patterns, which cannot be taken from a normal trashcan. This information can be used to predict and optimize some tasks in these spaces. Finally, we explored the potential of using blockchain technology to create economic incentives for recycling, following a Save-as-you-Throw (SAYT) model. 

**Abstract (ZH)**: 随着机器人系统（RS）变得更加自主，它们在小空间和办公室中被用于自动化清洁、基础设施维护或资源管理等任务。本文提出了一种名为iTrash的智能垃圾桶，旨在提高小办公室空间内的回收率。我们进行了一项为期5天的实验，发现iTrash相较于传统垃圾桶可以提高效率超过30%。本文研究结果表明，使用iTrash不仅能提高回收率，还能提供有价值的数据，如用户行为或垃圾桶使用模式，这些信息无法从普通垃圾桶中获取。这些信息可用于预测和优化这些空间内的某些任务。最后，我们探讨了利用区块链技术创建基于“边扔边省”（Save-as-you-Throw，SAYT）模型的回收经济激励机制的潜在可能性。 

---
# Monitoring snow avalanches from SAR data with deep learning 

**Title (ZH)**: 使用深度学习从SAR数据监测雪崩 

**Authors**: Filippo Maria Bianchi, Jakob Grahn  

**Link**: [PDF](https://arxiv.org/pdf/2502.18157)  

**Abstract**: Snow avalanches present significant risks to human life and infrastructure, particularly in mountainous regions, making effective monitoring crucial. Traditional monitoring methods, such as field observations, are limited by accessibility, weather conditions, and cost. Satellite-borne Synthetic Aperture Radar (SAR) data has become an important tool for large-scale avalanche detection, as it can capture data in all weather conditions and across remote areas. However, traditional processing methods struggle with the complexity and variability of avalanches. This chapter reviews the application of deep learning for detecting and segmenting snow avalanches from SAR data. Early efforts focused on the binary classification of SAR images, while recent advances have enabled pixel-level segmentation, providing greater accuracy and spatial resolution. A case study using Sentinel-1 SAR data demonstrates the effectiveness of deep learning models for avalanche segmentation, achieving superior results over traditional methods. We also present an extension of this work, testing recent state-of-the-art segmentation architectures on an expanded dataset of over 4,500 annotated SAR images. The best-performing model among those tested was applied for large-scale avalanche detection across the whole of Norway, revealing important spatial and temporal patterns over several winter seasons. 

**Abstract (ZH)**: 雷达合成孔径卫星数据中基于深度学习的雪崩检测与分割研究 

---
# Can LLMs Explain Themselves Counterfactually? 

**Title (ZH)**: LLMs能否进行反事实解释？ 

**Authors**: Zahra Dehghanighobadi, Asja Fischer, Muhammad Bilal Zafar  

**Link**: [PDF](https://arxiv.org/pdf/2502.18156)  

**Abstract**: Explanations are an important tool for gaining insights into the behavior of ML models, calibrating user trust and ensuring regulatory compliance. Past few years have seen a flurry of post-hoc methods for generating model explanations, many of which involve computing model gradients or solving specially designed optimization problems. However, owing to the remarkable reasoning abilities of Large Language Model (LLMs), self-explanation, that is, prompting the model to explain its outputs has recently emerged as a new paradigm. In this work, we study a specific type of self-explanations, self-generated counterfactual explanations (SCEs). We design tests for measuring the efficacy of LLMs in generating SCEs. Analysis over various LLM families, model sizes, temperature settings, and datasets reveals that LLMs sometimes struggle to generate SCEs. Even when they do, their prediction often does not agree with their own counterfactual reasoning. 

**Abstract (ZH)**: 自解释：大型语言模型生成-counterfactual解释的研究 

---
# SASSHA: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation 

**Title (ZH)**: SASSHA: 尖锐感知自适应二阶优化与稳定海森矩阵逼近 

**Authors**: Dahun Shin, Dongyeop Lee, Jinseok Chung, Namhoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.18153)  

**Abstract**: Approximate second-order optimization methods often exhibit poorer generalization compared to first-order approaches. In this work, we look into this issue through the lens of the loss landscape and find that existing second-order methods tend to converge to sharper minima compared to SGD. In response, we propose Sassha, a novel second-order method designed to enhance generalization by explicitly reducing sharpness of the solution, while stabilizing the computation of approximate Hessians along the optimization trajectory. In fact, this sharpness minimization scheme is crafted also to accommodate lazy Hessian updates, so as to secure efficiency besides flatness. To validate its effectiveness, we conduct a wide range of standard deep learning experiments where Sassha demonstrates its outstanding generalization performance that is comparable to, and mostly better than, other methods. We provide a comprehensive set of analyses including convergence, robustness, stability, efficiency, and cost. 

**Abstract (ZH)**: 逼近二次优化方法在泛化能力上往往逊于一阶方法。本文从损失景观的角度探讨这一问题，发现现有的二次优化方法倾向于收敛到比SGD更尖锐的极小值。为此，我们提出Sassha，一种新型的二次优化方法，旨在通过显式地减少解的尖锐性来增强泛化能力，同时在优化轨迹中稳定近似海塞矩阵的计算。实际上，该尖锐性最小化方案还设计了懒惰的海塞矩阵更新机制，以确保效率和平坦性。为了验证其效果，我们在一系列标准深度学习实验中测试了Sassha，结果显示其泛化性能与甚至优于其他方法。我们提供了包括收敛性、鲁棒性、稳定性、效率和成本在内的全面分析。 

---
# A Real-time Spatio-Temporal Trajectory Planner for Autonomous Vehicles with Semantic Graph Optimization 

**Title (ZH)**: 基于语义图优化的实时时空轨迹规划算法 

**Authors**: Shan He, Yalong Ma, Tao Song, Yongzhi Jiang, Xinkai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18151)  

**Abstract**: Planning a safe and feasible trajectory for autonomous vehicles in real-time by fully utilizing perceptual information in complex urban environments is challenging. In this paper, we propose a spatio-temporal trajectory planning method based on graph optimization. It efficiently extracts the multi-modal information of the perception module by constructing a semantic spatio-temporal map through separation processing of static and dynamic obstacles, and then quickly generates feasible trajectories via sparse graph optimization based on a semantic spatio-temporal hypergraph. Extensive experiments have proven that the proposed method can effectively handle complex urban public road scenarios and perform in real time. We will also release our codes to accommodate benchmarking for the research community 

**Abstract (ZH)**: 基于图形优化的时空路径规划方法：在复杂城市环境中实时规划安全可行的自主车辆轨迹 

---
# Jacobian Sparse Autoencoders: Sparsify Computations, Not Just Activations 

**Title (ZH)**: Jacobian稀疏自编码器：使计算稀疏，而不仅仅是激活函数 

**Authors**: Lucy Farnik, Tim Lawson, Conor Houghton, Laurence Aitchison  

**Link**: [PDF](https://arxiv.org/pdf/2502.18147)  

**Abstract**: Sparse autoencoders (SAEs) have been successfully used to discover sparse and human-interpretable representations of the latent activations of LLMs. However, we would ultimately like to understand the computations performed by LLMs and not just their representations. The extent to which SAEs can help us understand computations is unclear because they are not designed to "sparsify" computations in any sense, only latent activations. To solve this, we propose Jacobian SAEs (JSAEs), which yield not only sparsity in the input and output activations of a given model component but also sparsity in the computation (formally, the Jacobian) connecting them. With a naïve implementation, the Jacobians in LLMs would be computationally intractable due to their size. One key technical contribution is thus finding an efficient way of computing Jacobians in this setup. We find that JSAEs extract a relatively large degree of computational sparsity while preserving downstream LLM performance approximately as well as traditional SAEs. We also show that Jacobians are a reasonable proxy for computational sparsity because MLPs are approximately linear when rewritten in the JSAE basis. Lastly, we show that JSAEs achieve a greater degree of computational sparsity on pre-trained LLMs than on the equivalent randomized LLM. This shows that the sparsity of the computational graph appears to be a property that LLMs learn through training, and suggests that JSAEs might be more suitable for understanding learned transformer computations than standard SAEs. 

**Abstract (ZH)**: Jacobian 自编码器 (JSAEs) 用于发现 LLMs 的紧凑且计算可解释的表示 

---
# Large Language Model Driven Agents for Simulating Echo Chamber Formation 

**Title (ZH)**: 大型语言模型驱动的代理模拟回声室效应形成 

**Authors**: Chenhao Gu, Ling Luo, Zainab Razia Zaidi, Shanika Karunasekera  

**Link**: [PDF](https://arxiv.org/pdf/2502.18138)  

**Abstract**: The rise of echo chambers on social media platforms has heightened concerns about polarization and the reinforcement of existing beliefs. Traditional approaches for simulating echo chamber formation have often relied on predefined rules and numerical simulations, which, while insightful, may lack the nuance needed to capture complex, real-world interactions. In this paper, we present a novel framework that leverages large language models (LLMs) as generative agents to simulate echo chamber dynamics within social networks. The novelty of our approach is that it incorporates both opinion updates and network rewiring behaviors driven by LLMs, allowing for a context-aware and semantically rich simulation of social interactions. Additionally, we utilize real-world Twitter (now X) data to benchmark the LLM-based simulation against actual social media behaviors, providing insights into the accuracy and realism of the generated opinion trends. Our results demonstrate the efficacy of LLMs in modeling echo chamber formation, capturing both structural and semantic dimensions of opinion clustering. %This work contributes to a deeper understanding of social influence dynamics and offers a new tool for studying polarization in online communities. 

**Abstract (ZH)**: 社交媒体平台上的回声室效应兴起加剧了对 polarization 和现有信念强化的关注。本文提出了一种新颖框架，利用大规模语言模型（LLMs）作为生成代理来模拟社交网络内的回声室动态。该方法的 novelty 在于通过 LLMs 驱动意见更新和网络重构行为，实现一种具有上下文感知和语义丰富性的社交互动模拟。此外，我们使用真实的 Twitter（现 X）数据来验证基于 LLM 的模拟与实际社交媒体行为的契合度，提供生成意见趋势准确性和真实性的洞见。研究结果表明，LLMs 在建模回声室效应方面是有效的，能够捕捉意见簇的结构和语义维度。%本文加深了对社会影响动态的理解，并提供了一个研究在线社区极化的新工具。 

---
# SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inference 

**Title (ZH)**: SpargeAttn: 准确的稀疏注意力加速任意模型推断 

**Authors**: Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, Jianfei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18137)  

**Abstract**: An efficient attention implementation is essential for large models due to its quadratic time complexity. Fortunately, attention commonly exhibits sparsity, i.e., many values in the attention map are near zero, allowing for the omission of corresponding computations. Many studies have utilized the sparse pattern to accelerate attention. However, most existing works focus on optimizing attention within specific models by exploiting certain sparse patterns of the attention map. A universal sparse attention that guarantees both the speedup and end-to-end performance of diverse models remains elusive. In this paper, we propose SpargeAttn, a universal sparse and quantized attention for any model. Our method uses a two-stage online filter: in the first stage, we rapidly and accurately predict the attention map, enabling the skip of some matrix multiplications in attention. In the second stage, we design an online softmax-aware filter that incurs no extra overhead and further skips some matrix multiplications. Experiments show that our method significantly accelerates diverse models, including language, image, and video generation, without sacrificing end-to-end metrics. The codes are available at this https URL. 

**Abstract (ZH)**: 一种高效的注意力实现对于大型模型至关重要，因为其时间复杂度为二次。幸运的是，注意力通常表现出稀疏性，即注意力图中的许多值接近零，允许省略相应的计算。许多研究利用稀疏性来加速注意力机制。然而，大多数现有工作集中在通过利用注意力图的特定稀疏模式来优化特定模型的注意力机制上。一个同时保证多种模型加速和端到端性能的通用稀疏注意力机制仍然难以捉摸。本文提出了一种适用于任何模型的通用稀疏和量化注意力机制SpargeAttn。我们的方法使用两阶段在线过滤器：第一阶段我们快速准确地预测注意力图，从而省略一些矩阵乘法；第二阶段我们设计了一种在线Softmax感知过滤器，无需额外开销并进一步省略一些矩阵乘法。实验表明，我们的方法能够在不牺牲端到端指标的情况下显著加速包括语言、图像和视频生成等多种模型。源代码可在以下网址获取。 

---
# EU-Nets: Enhanced, Explainable and Parsimonious U-Nets 

**Title (ZH)**: EU-网络：增强、可解释且简洁的U-网络 

**Authors**: B. Sun, P. Liò  

**Link**: [PDF](https://arxiv.org/pdf/2502.18122)  

**Abstract**: In this study, we propose MHEX+, a framework adaptable to any U-Net architecture. Built upon MHEX+, we introduce novel U-Net variants, EU-Nets, which enhance explainability and uncertainty estimation, addressing the limitations of traditional U-Net models while improving performance and stability. A key innovation is the Equivalent Convolutional Kernel, which unifies consecutive convolutional layers, boosting interpretability. For uncertainty estimation, we propose the collaboration gradient approach, measuring gradient consistency across decoder layers. Notably, EU-Nets achieve an average accuracy improvement of 1.389\% and a variance reduction of 0.83\% across all networks and datasets in our experiments, requiring fewer than 0.1M parameters. 

**Abstract (ZH)**: 本研究提出MHEX+，一种适用于任何U-Net架构的框架。基于MHEX+，我们引入了增强可解释性和不确定性估计的新型U-Net变体EU-Nets，克服了传统U-Net模型的局限性，同时提高了性能和稳定性。关键创新是等价卷积核，它统一了连续卷积层，提升了解释性。对于不确定性估计，我们提出了一种合作梯度方法，衡量解码器层之间的梯度一致性。值得注意的是，EU-Nets在我们的实验中实现了所有网络和数据集的平均准确性提升1.389%和方差减少0.83%，同时所需参数少于0.1M。 

---
# Bayesian Optimization for Controlled Image Editing via LLMs 

**Title (ZH)**: 基于LLMs的可控图像编辑的贝叶斯优化 

**Authors**: Chengkun Cai, Haoliang Liu, Xu Zhao, Zhongyu Jiang, Tianfang Zhang, Zongkai Wu, Jenq-Neng Hwang, Serge Belongie, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.18116)  

**Abstract**: In the rapidly evolving field of image generation, achieving precise control over generated content and maintaining semantic consistency remain significant limitations, particularly concerning grounding techniques and the necessity for model fine-tuning. To address these challenges, we propose BayesGenie, an off-the-shelf approach that integrates Large Language Models (LLMs) with Bayesian Optimization to facilitate precise and user-friendly image editing. Our method enables users to modify images through natural language descriptions without manual area marking, while preserving the original image's semantic integrity. Unlike existing techniques that require extensive pre-training or fine-tuning, our approach demonstrates remarkable adaptability across various LLMs through its model-agnostic design. BayesGenie employs an adapted Bayesian optimization strategy to automatically refine the inference process parameters, achieving high-precision image editing with minimal user intervention. Through extensive experiments across diverse scenarios, we demonstrate that our framework significantly outperforms existing methods in both editing accuracy and semantic preservation, as validated using different LLMs including Claude3 and GPT-4. 

**Abstract (ZH)**: 在图像生成快速发展的领域中，实现对生成内容的精确控制和保持语义一致性仍然是重要的限制，特别是在 grounding 技术和模型微调的必要性方面。为了解决这些挑战，我们提出了 BayesGenie，一种将大型语言模型（LLMs）与贝叶斯优化相结合的现成方法，以促进精确和用户友好的图像编辑。该方法允许用户通过自然语言描述来修改图像而不进行手动区域标记，同时保留原始图像的语义完整性。与需要大量预训练或微调的现有技术不同，我们的方法通过其模型无感知的设计展示了在各种 LLMs 上的显著适应性。BayesGenie 采用适应性的贝叶斯优化策略自动细化推理过程参数，实现高精度的图像编辑并减少用户干预。通过在多样场景下的广泛实验，我们证明了我们的框架在编辑准确性和语义保真度方面显著优于现有方法，并得到了包括 Claude3 和 GPT-4 在内的不同 LLMs 的验证。 

---
# The Built-In Robustness of Decentralized Federated Averaging to Bad Data 

**Title (ZH)**: 内置鲁棒性的去中心化联邦平均对不良数据的抵抗能力 

**Authors**: Samuele Sabella, Chiara Boldrini, Lorenzo Valerio, Andrea Passarella, Marco Conti  

**Link**: [PDF](https://arxiv.org/pdf/2502.18097)  

**Abstract**: Decentralized federated learning (DFL) enables devices to collaboratively train models over complex network topologies without relying on a central controller. In this setting, local data remains private, but its quality and quantity can vary significantly across nodes. The extent to which a fully decentralized system is vulnerable to poor-quality or corrupted data remains unclear, but several factors could contribute to potential risks. Without a central authority, there can be no unified mechanism to detect or correct errors, and each node operates with a localized view of the data distribution, making it difficult for the node to assess whether its perspective aligns with the true distribution. Moreover, models trained on low-quality data can propagate through the network, amplifying errors. To explore the impact of low-quality data on DFL, we simulate two scenarios with degraded data quality -- one where the corrupted data is evenly distributed in a subset of nodes and one where it is concentrated on a single node -- using a decentralized implementation of FedAvg. Our results reveal that averaging-based decentralized learning is remarkably robust to localized bad data, even when the corrupted data resides in the most influential nodes of the network. Counterintuitively, this robustness is further enhanced when the corrupted data is concentrated on a single node, regardless of its centrality in the communication network topology. This phenomenon is explained by the averaging process, which ensures that no single node -- however central -- can disproportionately influence the overall learning process. 

**Abstract (ZH)**: 去中心化的联邦学习（DFL） enables 设备在无需依赖中心控制器的情况下，通过复杂网络拓扑协作训练模型。在这种设置中，本地数据保持私有，但其质量和数量在节点间可以有显著差异。完全去中心化系统受低质量或被篡改数据的影响程度仍然不清楚，但有几个因素可能增加潜在风险。缺乏中央权威机构，就没有统一机制来检测或纠正错误，每个节点仅拥有局部数据分布视图，难以评估其视角是否与真实分布一致。此外，基于低质量数据训练的模型可以通过网络传播，放大错误。为了探索低质量数据对 DFL 的影响，我们使用去中心化的 FedAvg 实现模拟了两种数据质量降级场景——一种是被篡改数据均匀分布在一个子集节点中，另一种是被篡改数据集中在单个节点中。结果显示，基于平均值的去中心化学习对局部劣质数据表现出惊人的鲁棒性，即使被篡改数据位于网络中最具影响力的节点中也是如此。令人意外的是，当被篡改数据集中在单个节点中时，这种鲁棒性反而增强了，无论该节点在通信网络拓扑中的中心性如何。这种现象可通过平均过程解释，该过程确保没有单一节点——无论其多么中心——能够不成比例地影响整体学习过程。 

---
# Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning 

**Title (ZH)**: 面向推理最优测试时计算量的扩展研究 

**Authors**: Wenkai Yang, Shuming Ma, Yankai Lin, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.18080)  

**Abstract**: Recent studies have shown that making a model spend more time thinking through longer Chain of Thoughts (CoTs) enables it to gain significant improvements in complex reasoning tasks. While current researches continue to explore the benefits of increasing test-time compute by extending the CoT lengths of Large Language Models (LLMs), we are concerned about a potential issue hidden behind the current pursuit of test-time scaling: Would excessively scaling the CoT length actually bring adverse effects to a model's reasoning performance? Our explorations on mathematical reasoning tasks reveal an unexpected finding that scaling with longer CoTs can indeed impair the reasoning performance of LLMs in certain domains. Moreover, we discover that there exists an optimal scaled length distribution that differs across different domains. Based on these insights, we propose a Thinking-Optimal Scaling strategy. Our method first uses a small set of seed data with varying response length distributions to teach the model to adopt different reasoning efforts for deep thinking. Then, the model selects its shortest correct response under different reasoning efforts on additional problems for self-improvement. Our self-improved models built upon Qwen2.5-32B-Instruct outperform other distillation-based 32B o1-like models across various math benchmarks, and achieve performance on par with QwQ-32B-Preview. 

**Abstract (ZH)**: 最近的研究表明，让模型通过更长的Chain of Thoughts（CoTs）进行更多思考可以显著提高其在复杂推理任务中的表现。尽管当前研究继续探索通过延长大型语言模型（LLMs）的CoT长度来增加测试时计算量带来的益处，但我们担心当前追求测试时扩展背后隐藏的问题：过度扩展CoT长度是否会对模型的推理性能产生不利影响？我们的探索在数学推理任务中揭示了一个意外的发现：在某些领域，使用更长的CoTs进行扩展确实会损害LLMs的推理性能。此外，我们发现不同领域存在不同的最优扩展长度分布。基于这些洞察，我们提出了一种思考最优扩展策略。该方法首先使用具有不同响应长度分布的一组种子数据来教导模型根据不同推理努力进行深入思考。然后，模型在不同推理努力下选择其最短的正确响应进行自我改进。基于Qwen2.5-32B-Instruct构建的自我改进模型在各种数学基准测试中优于其他基于蒸馏的32B o1-like模型，并且性能与QwQ-32B-Preview相当。 

---
# MRBTP: Efficient Multi-Robot Behavior Tree Planning and Collaboration 

**Title (ZH)**: MRBTP：高效多机器人行为树规划与协作 

**Authors**: Yishuai Cai, Xinglin Chen, Zhongxuan Cai, Yunxin Mao, Minglong Li, Wenjing Yang, Ji Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18072)  

**Abstract**: Multi-robot task planning and collaboration are critical challenges in robotics. While Behavior Trees (BTs) have been established as a popular control architecture and are plannable for a single robot, the development of effective multi-robot BT planning algorithms remains challenging due to the complexity of coordinating diverse action spaces. We propose the Multi-Robot Behavior Tree Planning (MRBTP) algorithm, with theoretical guarantees of both soundness and completeness. MRBTP features cross-tree expansion to coordinate heterogeneous actions across different BTs to achieve the team's goal. For homogeneous actions, we retain backup structures among BTs to ensure robustness and prevent redundant execution through intention sharing. While MRBTP is capable of generating BTs for both homogeneous and heterogeneous robot teams, its efficiency can be further improved. We then propose an optional plugin for MRBTP when Large Language Models (LLMs) are available to reason goal-related actions for each robot. These relevant actions can be pre-planned to form long-horizon subtrees, significantly enhancing the planning speed and collaboration efficiency of MRBTP. We evaluate our algorithm in warehouse management and everyday service scenarios. Results demonstrate MRBTP's robustness and execution efficiency under varying settings, as well as the ability of the pre-trained LLM to generate effective task-specific subtrees for MRBTP. 

**Abstract (ZH)**: 多机器人行为树规划（MRBTP）及其在仓库管理和日常服务场景中的应用 

---
# HEROS-GAN: Honed-Energy Regularized and Optimal Supervised GAN for Enhancing Accuracy and Range of Low-Cost Accelerometers 

**Title (ZH)**: HEROS-GAN：精炼能量正则化和最优监督生成对抗网络以提升低成本加速度计的精度和量程 

**Authors**: Yifeng Wang, Yi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18064)  

**Abstract**: Low-cost accelerometers play a crucial role in modern society due to their advantages of small size, ease of integration, wearability, and mass production, making them widely applicable in automotive systems, aerospace, and wearable technology. However, this widely used sensor suffers from severe accuracy and range limitations. To this end, we propose a honed-energy regularized and optimal supervised GAN (HEROS-GAN), which transforms low-cost sensor signals into high-cost equivalents, thereby overcoming the precision and range limitations of low-cost accelerometers. Due to the lack of frame-level paired low-cost and high-cost signals for training, we propose an Optimal Transport Supervision (OTS), which leverages optimal transport theory to explore potential consistency between unpaired data, thereby maximizing supervisory information. Moreover, we propose a Modulated Laplace Energy (MLE), which injects appropriate energy into the generator to encourage it to break range limitations, enhance local changes, and enrich signal details. Given the absence of a dedicated dataset, we specifically establish a Low-cost Accelerometer Signal Enhancement Dataset (LASED) containing tens of thousands of samples, which is the first dataset serving to improve the accuracy and range of accelerometers and is released in Github. Experimental results demonstrate that a GAN combined with either OTS or MLE alone can surpass the previous signal enhancement SOTA methods by an order of magnitude. Integrating both OTS and MLE, the HEROS-GAN achieves remarkable results, which doubles the accelerometer range while reducing signal noise by two orders of magnitude, establishing a benchmark in the accelerometer signal processing. 

**Abstract (ZH)**: 低成本加速度计在现代社会中扮演着至关重要的角色，得益于其小型化、易于集成、可穿戴和大规模生产的优势，使其广泛应用于汽车系统、航空航天和可穿戴技术。然而，这种广泛使用的传感器面临着严重的精度和量程限制。为此，我们提出了一种精炼能量正则化和最优监督生成对抗网络（HEROS-GAN），该方法将低成本传感器信号转化为高精度等效信号，从而克服了低成本加速度计的精度和量程限制。由于缺乏帧级配对的低成本和高成本信号进行训练，我们提出了最优运输监督（OTS），利用最优运输理论来探索未配对数据之间的潜在一致性，从而最大化监督信息。此外，我们提出了一种调制拉普拉斯能量（MLE），向生成器注入适当的能量，促使生成器打破量程限制，增强局部变化并丰富信号细节。鉴于缺乏专用数据集，我们特别建立了包含数万个样本的低成本加速度计信号增强数据集（LASED），这是首个用于提高加速度计精度和量程的数据集，并发布在Github上。实验结果表明，单独使用OTS或MLE的GAN都能比之前最先进的信号增强方法优越一个数量级。结合OTS和MLE的HEROS-GAN取得了显著成果，加速度计的量程翻倍，信号噪声降低两个数量级，建立了加速度计信号处理的基准。 

---
# VLM-E2E: Enhancing End-to-End Autonomous Driving with Multimodal Driver Attention Fusion 

**Title (ZH)**: VLM-E2E: 基于多模态驾驶员注意力融合的端到端自动驾驶增强 

**Authors**: Pei Liu, Haipeng Liu, Haichao Liu, Xin Liu, Jinxin Ni, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.18042)  

**Abstract**: Human drivers adeptly navigate complex scenarios by utilizing rich attentional semantics, but the current autonomous systems struggle to replicate this ability, as they often lose critical semantic information when converting 2D observations into 3D space. In this sense, it hinders their effective deployment in dynamic and complex environments. Leveraging the superior scene understanding and reasoning abilities of Vision-Language Models (VLMs), we propose VLM-E2E, a novel framework that uses the VLMs to enhance training by providing attentional cues. Our method integrates textual representations into Bird's-Eye-View (BEV) features for semantic supervision, which enables the model to learn richer feature representations that explicitly capture the driver's attentional semantics. By focusing on attentional semantics, VLM-E2E better aligns with human-like driving behavior, which is critical for navigating dynamic and complex environments. Furthermore, we introduce a BEV-Text learnable weighted fusion strategy to address the issue of modality importance imbalance in fusing multimodal information. This approach dynamically balances the contributions of BEV and text features, ensuring that the complementary information from visual and textual modality is effectively utilized. By explicitly addressing the imbalance in multimodal fusion, our method facilitates a more holistic and robust representation of driving environments. We evaluate VLM-E2E on the nuScenes dataset and demonstrate its superiority over state-of-the-art approaches, showcasing significant improvements in performance. 

**Abstract (ZH)**: 利用视觉语言模型的优越场景理解与推理能力，我们提出了一种名为VLM-E2E的新框架，通过提供注意力线索来增强训练。该方法将文本表示整合到Bird's-Eye-View (BEV)特征中，进行语义监督，使模型能够学习更丰富的特征表示，明确捕捉驾驶者的注意力语义。通过关注注意力语义，VLM-E2E更好地与类似人类的驾驶行为对齐，这对于导航动态和复杂的环境至关重要。此外，我们引入了一种BEV-Text可学习加权融合策略，以解决多模态信息融合中模态重要性不平衡的问题。该方法动态平衡BEV和文本特征的贡献，确保视觉和文本模态互补信息的有效利用。通过明确解决多模态融合中的不平衡问题，我们的方法促进了对驾驶环境更全面和 robust 的表示。我们在nuScenes数据集上评估了VLM-E2E，并展示了其在性能上的优越性，表现出显著的改进。 

---
# AutoCas: Autoregressive Cascade Predictor in Social Networks via Large Language Models 

**Title (ZH)**: AutoCas: 社交网络中基于大规模语言模型的自回归级联预测器 

**Authors**: Yuhao Zheng, Chenghua Gong, Rui Sun, Juyuan Zhang, Liming Pan, Linyuan Lv  

**Link**: [PDF](https://arxiv.org/pdf/2502.18040)  

**Abstract**: Popularity prediction in information cascades plays a crucial role in social computing, with broad applications in viral marketing, misinformation control, and content recommendation. However, information propagation mechanisms, user behavior, and temporal activity patterns exhibit significant diversity, necessitating a foundational model capable of adapting to such variations. At the same time, the amount of available cascade data remains relatively limited compared to the vast datasets used for training large language models (LLMs). Recent studies have demonstrated the feasibility of leveraging LLMs for time-series prediction by exploiting commonalities across different time-series domains. Building on this insight, we introduce the Autoregressive Information Cascade Predictor (AutoCas), an LLM-enhanced model designed specifically for cascade popularity prediction. Unlike natural language sequences, cascade data is characterized by complex local topologies, diffusion contexts, and evolving dynamics, requiring specialized adaptations for effective LLM integration. To address these challenges, we first tokenize cascade data to align it with sequence modeling principles. Next, we reformulate cascade diffusion as an autoregressive modeling task to fully harness the architectural strengths of LLMs. Beyond conventional approaches, we further introduce prompt learning to enhance the synergy between LLMs and cascade prediction. Extensive experiments demonstrate that AutoCas significantly outperforms baseline models in cascade popularity prediction while exhibiting scaling behavior inherited from LLMs. Code is available at this repository: this https URL 

**Abstract (ZH)**: 信息 cascade 中的流行度预测在社会计算中起着关键作用，广泛应用于病毒营销、虚假信息控制和内容推荐。然而，信息传播机制、用户行为和时间活动模式表现出显著的多样性，需要一种能适应这些变化的基礎模型。同时，可用的 cascade 数据量相对有限，远少于用于训练大型语言模型（LLMs）的大规模数据集。近期研究表明，通过利用不同时间序列领域的共性，可以利用 LLMs 进行时间序列预测。基于这一认识，我们提出了 Autoregressive Information Cascade Predictor (AutoCas)，一种增强的 LLM 模型，专门用于 cascade 流行度预测。不同于自然语言序列，cascade 数据特征复杂，包含复杂的地方拓扑、传播上下文和演变动力学，需要专门的适应以有效集成 LLMs。为了解决这些挑战，我们首先对 cascade 数据进行分词，使其与序列建模原则对齐。接着，我们将 cascade 传播重新表述为一个自回归建模任务，以充分利用 LLMs 的架构优势。除此之外，我们还引入了提示学习以增强 LLMs 和 cascade 预测之间的协同作用。广泛的实验表明，AutoCas 在 cascade 流行度预测方面显著优于基准模型，并且表现出继承自 LLMs 的可扩展性。代码可在以下仓库获取：this https URL。 

---
# ExPath: Towards Explaining Targeted Pathways for Biological Knowledge Bases 

**Title (ZH)**: ExPath：面向生物知识库的路径解释方法研究 

**Authors**: Rikuto Kotoge, Ziwei Yang, Zheng Chen, Yushun Dong, Yasuko Matsubara, Jimeng Sun, Yasushi Sakurai  

**Link**: [PDF](https://arxiv.org/pdf/2502.18026)  

**Abstract**: Biological knowledge bases provide systemically functional pathways of cells or organisms in terms of molecular interaction. However, recognizing more targeted pathways, particularly when incorporating wet-lab experimental data, remains challenging and typically requires downstream biological analyses and expertise. In this paper, we frame this challenge as a solvable graph learning and explaining task and propose a novel pathway inference framework, ExPath, that explicitly integrates experimental data, specifically amino acid sequences (AA-seqs), to classify various graphs (bio-networks) in biological databases. The links (representing pathways) that contribute more to classification can be considered as targeted pathways. Technically, ExPath comprises three components: (1) a large protein language model (pLM) that encodes and embeds AA-seqs into graph, overcoming traditional obstacles in processing AA-seq data, such as BLAST; (2) PathMamba, a hybrid architecture combining graph neural networks (GNNs) with state-space sequence modeling (Mamba) to capture both local interactions and global pathway-level dependencies; and (3) PathExplainer, a subgraph learning module that identifies functionally critical nodes and edges through trainable pathway masks. We also propose ML-oriented biological evaluations and a new metric. The experiments involving 301 bio-networks evaluations demonstrate that pathways inferred by ExPath maintain biological meaningfulness. We will publicly release curated 301 bio-network data soon. 

**Abstract (ZH)**: 生物知识库提供了基于分子相互作用的细胞或 organism 的系统功能路径。然而，在结合湿实验室实验数据的情况下识别更具针对性的路径仍具有挑战性，通常需要下游生物学分析和专业知识。本文将这一挑战视为可解决的图学习和解释任务，并提出了一种新的路径推理框架 ExPath，该框架明确整合了实验数据，特别是氨基酸序列（AA-seqs），以对生物数据库中的各种图（生物网络）进行分类。对分类贡献更大的连接可以视为针对性路径。技术上，ExPath 包含三个组成部分：（1）一个大规模蛋白质语言模型（pLM），它将氨基酸序列编码并嵌入图中，克服了处理氨基酸序列数据的传统障碍，如 BLAST；（2）PathMamba，这是一种将图神经网络（GNNs）与状态空间序列建模（Mamba）结合的混合架构，用于捕获局部相互作用和全局路径级依赖性；（3）PathExplainer，这是一种子图学习模块，通过可训练的路径掩码识别功能关键节点和边。我们还提出了面向机器学习的生物评价和一个新的度量标准。涉及 301 个生物网络的实验表明，ExPath 推断出的路径具有生物学意义。我们很快将公开发布 301 个生物网络数据。 

---
# AfroXLMR-Comet: Multilingual Knowledge Distillation with Attention Matching for Low-Resource languages 

**Title (ZH)**: AfroXLMR-Comet: 基于注意力匹配的多语言知识蒸馏方法及其在低资源语言中的应用 

**Authors**: Joshua Sakthivel Raju, Sanjay S, Jaskaran Singh Walia, Srinivas Raghav, Vukosi Marivate  

**Link**: [PDF](https://arxiv.org/pdf/2502.18020)  

**Abstract**: Language model compression through knowledge distillation has emerged as a promising approach for deploying large language models in resource-constrained environments. However, existing methods often struggle to maintain performance when distilling multilingual models, especially for low-resource languages. In this paper, we present a novel hybrid distillation approach that combines traditional knowledge distillation with a simplified attention matching mechanism, specifically designed for multilingual contexts. Our method introduces an extremely compact student model architecture, significantly smaller than conventional multilingual models. We evaluate our approach on five African languages: Kinyarwanda, Swahili, Hausa, Igbo, and Yoruba. The distilled student model; AfroXLMR-Comet successfully captures both the output distribution and internal attention patterns of a larger teacher model (AfroXLMR-Large) while reducing the model size by over 85%. Experimental results demonstrate that our hybrid approach achieves competitive performance compared to the teacher model, maintaining an accuracy within 85% of the original model's performance while requiring substantially fewer computational resources. Our work provides a practical framework for deploying efficient multilingual models in resource-constrained environments, particularly benefiting applications involving African languages. 

**Abstract (ZH)**: 通过知识蒸馏的语言模型压缩在资源受限环境中部署大规模语言模型已 emerges as a promising approach.然而，现有方法在蒸馏多语言模型时往往难以保持性能，尤其是对于低资源语言。在本文中，我们提出了一种新颖的混合蒸馏方法，结合了传统的知识蒸馏和简化后的注意力匹配机制，特别适用于多语言环境。我们的方法引入了一种极其紧凑的学生模型架构，明显小于传统的多语言模型。我们在五种非洲语言： Kirundi、Swahili、Hausa、Igbo 和 Yoruba 上评估了我们的方法。蒸馏后的学生模型 AfroXLMR-Comet 成功地捕捉到了更大教师模型（AfroXLMR-Large）的输出分布和内部注意力模式，同时将模型大小减少了超过 85%。实验结果表明，我们的混合方法在性能上与教师模型相当，在计算资源需求显著减少的情况下，准确率达到了原模型性能的 85% 以内。我们的工作提供了一种实用的框架，用于在资源受限环境中部署高效的多语言模型，特别有利于涉及非洲语言的应用。 

---
# ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents 

**Title (ZH)**: 视觉文档检索增强生成 via 动态迭代推理代理 

**Authors**: Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shihang Wang, Pengjun Xie, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18017)  

**Abstract**: Understanding information from visually rich documents remains a significant challenge for traditional Retrieval-Augmented Generation (RAG) methods. Existing benchmarks predominantly focus on image-based question answering (QA), overlooking the fundamental challenges of efficient retrieval, comprehension, and reasoning within dense visual documents. To bridge this gap, we introduce ViDoSeek, a novel dataset designed to evaluate RAG performance on visually rich documents requiring complex reasoning. Based on it, we identify key limitations in current RAG approaches: (i) purely visual retrieval methods struggle to effectively integrate both textual and visual features, and (ii) previous approaches often allocate insufficient reasoning tokens, limiting their effectiveness. To address these challenges, we propose ViDoRAG, a novel multi-agent RAG framework tailored for complex reasoning across visual documents. ViDoRAG employs a Gaussian Mixture Model (GMM)-based hybrid strategy to effectively handle multi-modal retrieval. To further elicit the model's reasoning capabilities, we introduce an iterative agent workflow incorporating exploration, summarization, and reflection, providing a framework for investigating test-time scaling in RAG domains. Extensive experiments on ViDoSeek validate the effectiveness and generalization of our approach. Notably, ViDoRAG outperforms existing methods by over 10% on the competitive ViDoSeek benchmark. 

**Abstract (ZH)**: 理解视觉丰富文档中的信息仍然是传统检索增强生成（RAG）方法的一项重大挑战。现有的基准主要集中在基于图像的问答（QA），忽视了密集视觉文档中高效检索、理解和推理的基本挑战。为了弥补这一差距，我们引入了ViDoSeek，一个旨在评估RAG在需要复杂推理的视觉丰富文档上的性能的新颖数据集。基于此，我们识别出当前RAG方法的关键局限性：（i）纯粹基于视觉的检索方法难以有效整合文本和视觉特征，（ii）先前的方法通常分配了不足的推理令牌，限制了它们的有效性。为了应对这些挑战，我们提出了ViDoRAG，一种针对视觉文档上复杂推理的新型多代理RAG框架。ViDoRAG采用基于高斯混合模型（GMM）的混合策略，有效地处理多模态检索。为了进一步激发模型的推理能力，我们引入了一种迭代的代理工作流，包括探索、总结和反思，提供了一个框架用于在RAG领域研究测试时标度问题。广泛的ViDoSeek实验验证了我们方法的有效性和普适性。值得注意的是，ViDoRAG在竞争性的ViDoSeek基准上性能优于现有方法，超过10%。 

---
# NotaGen: Advancing Musicality in Symbolic Music Generation with Large Language Model Training Paradigms 

**Title (ZH)**: NotaGen：大规模语言模型训练范式在符号音乐生成中提升音乐性 

**Authors**: Yashan Wang, Shangda Wu, Jianhuai Hu, Xingjian Du, Yueqi Peng, Yongxin Huang, Shuai Fan, Xiaobing Li, Feng Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.18008)  

**Abstract**: We introduce NotaGen, a symbolic music generation model aiming to explore the potential of producing high-quality classical sheet music. Inspired by the success of Large Language Models (LLMs), NotaGen adopts pre-training, fine-tuning, and reinforcement learning paradigms (henceforth referred to as the LLM training paradigms). It is pre-trained on 1.6M pieces of music, and then fine-tuned on approximately 9K high-quality classical compositions conditioned on "period-composer-instrumentation" prompts. For reinforcement learning, we propose the CLaMP-DPO method, which further enhances generation quality and controllability without requiring human annotations or predefined rewards. Our experiments demonstrate the efficacy of CLaMP-DPO in symbolic music generation models with different architectures and encoding schemes. Furthermore, subjective A/B tests show that NotaGen outperforms baseline models against human compositions, greatly advancing musical aesthetics in symbolic music this http URL project homepage is this https URL. 

**Abstract (ZH)**: NotaGen：一种探索高质量古典乐谱生成潜力的符号化音乐生成模型 

---
# Radon-Nikodým Derivative: Re-imagining Anomaly Detection from a Measure Theoretic Perspective 

**Title (ZH)**: Radon-Nikodým 导数：从测度论角度重构异常检测 

**Authors**: Shlok Mehendale, Aditya Challa, Rahul Yedida, Sravan Danda, Santonu Sarkar, Snehanshu Saha  

**Link**: [PDF](https://arxiv.org/pdf/2502.18002)  

**Abstract**: Which principle underpins the design of an effective anomaly detection loss function? The answer lies in the concept of \rnthm{} theorem, a fundamental concept in measure theory. The key insight is -- Multiplying the vanilla loss function with the \rnthm{} derivative improves the performance across the board. We refer to this as RN-Loss. This is established using PAC learnability of anomaly detection. We further show that the \rnthm{} derivative offers important insights into unsupervised clustering based anomaly detections as well. We evaluate our algorithm on 96 datasets, including univariate and multivariate data from diverse domains, including healthcare, cybersecurity, and finance. We show that RN-Derivative algorithms outperform state-of-the-art methods on 68\% of Multivariate datasets (based on F-1 scores) and also achieves peak F1-scores on 72\% of time series (Univariate) datasets. 

**Abstract (ZH)**: 哪种原则支撑了有效异常检测损失函数的设计？答案在于测度论中的\ rnthm{}定理。关键见解在于——将标准损失函数与\ rnthm{}导数相乘可以全面提升性能。我们称此为RN-Loss。我们使用PAC学习能力来确立这一点。进一步研究表明，\ rnthm{}导数还为基于无监督聚类的异常检测提供了重要的见解。我们在96个数据集上评估了我们的算法，包括来自健康医疗、网络安全和金融等多个领域的单变量和多变量数据。结果显示，基于F-1分数，RN-导数算法在68%的多变量数据集上优于现有最佳方法，并在72%的时间序列（单变量）数据集上实现了最高的F1分数。 

---
# MAGE: Multi-Head Attention Guided Embeddings for Low Resource Sentiment Classification 

**Title (ZH)**: MAGE: 多头注意力引导嵌入低资源情感分类 

**Authors**: Varun Vashisht, Samar Singh, Mihir Konduskar, Jaskaran Singh Walia, Vukosi Marivate  

**Link**: [PDF](https://arxiv.org/pdf/2502.17987)  

**Abstract**: Due to the lack of quality data for low-resource Bantu languages, significant challenges are presented in text classification and other practical implementations. In this paper, we introduce an advanced model combining Language-Independent Data Augmentation (LiDA) with Multi-Head Attention based weighted embeddings to selectively enhance critical data points and improve text classification performance. This integration allows us to create robust data augmentation strategies that are effective across various linguistic contexts, ensuring that our model can handle the unique syntactic and semantic features of Bantu languages. This approach not only addresses the data scarcity issue but also sets a foundation for future research in low-resource language processing and classification tasks. 

**Abstract (ZH)**: 由于缺乏高质量的数据资源，Bantu低资源语言在文本分类和其他实际应用中面临重大挑战。本文介绍了一种结合了语言独立数据增强（LiDA）与多头注意力加权嵌入的先进模型，以选择性地增强关键数据点并提高文本分类性能。这种集成方案使我们能够创建适用于各种语言背景的稳健数据增强策略，确保模型能够处理Bantu语言的特殊句法和语义特征。这种方法不仅解决了数据稀缺问题，也为未来低资源语言处理和分类任务的研究奠定了基础。 

---
# Broadening Discovery through Structural Models: Multimodal Combination of Local and Structural Properties for Predicting Chemical Features 

**Title (ZH)**: 通过结构模型扩展发现：结合局部和结构属性的多模态组合预测化学特征 

**Authors**: Nikolai Rekut, Alexey Orlov, Klea Ziu, Elizaveta Starykh, Martin Takac, Aleksandr Beznosikov  

**Link**: [PDF](https://arxiv.org/pdf/2502.17986)  

**Abstract**: In recent years, machine learning has profoundly reshaped the field of chemistry, facilitating significant advancements across various applications, including the prediction of molecular properties and the generation of molecular structures. Language models and graph-based models are extensively utilized within this domain, consistently achieving state-of-the-art results across an array of tasks. However, the prevailing practice of representing chemical compounds in the SMILES format -- used by most datasets and many language models -- presents notable limitations as a training data format. In contrast, chemical fingerprints offer a more physically informed representation of compounds, thereby enhancing their suitability for model training. This study aims to develop a language model that is specifically trained on fingerprints. Furthermore, we introduce a bimodal architecture that integrates this language model with a graph model. Our proposed methodology synthesizes these approaches, utilizing RoBERTa as the language model and employing Graph Isomorphism Networks (GIN), Graph Convolutional Networks (GCN) and Graphormer as graph models. This integration results in a significant improvement in predictive performance compared to conventional strategies for tasks such as Quantitative Structure-Activity Relationship (QSAR) and the prediction of nuclear magnetic resonance (NMR) spectra, among others. 

**Abstract (ZH)**: 近年来，机器学习深刻重塑了化学领域，推动了分子性质预测和分子结构生成等各类应用的重要进展。语言模型和图基模型在此领域被广泛使用，始终在众多任务上取得最先进成果。然而，大多数数据集和语言模型使用的SMILES表示格式作为训练数据格式存在明显局限性。相比之下，化学指纹提供了更物理化的化合物表示，从而增强了其作为模型训练数据的适用性。本研究旨在开发一种专门在指纹上训练的语言模型，并引入了一种双模架构，将这种语言模型与图模型结合。我们提出的方法综合了这些方法，使用RoBERTa作为语言模型，并采用Graph Isomorphism Networks (GIN)、Graph Convolutional Networks (GCN) 和Graphormer作为图模型。这种集成在定量结构-活性关系（QSAR）和核磁共振（NMR）谱预测等任务上显著提升了预测性能。 

---
# LLM Knows Geometry Better than Algebra: Numerical Understanding of LLM-Based Agents in A Trading Arena 

**Title (ZH)**: LLM在交易 arena 中对几何的理解优于代数：基于LLM的智能体的数值理解 

**Authors**: Tianmi Ma, Jiawei Du, Wenxin Huang, Wenjie Wang, Liang Xie, Xian Zhong, Joey Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.17967)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly improved performance in natural language processing tasks. However, their ability to generalize to dynamic, unseen tasks, particularly in numerical reasoning, remains a challenge. Existing benchmarks mainly evaluate LLMs on problems with predefined optimal solutions, which may not align with real-world scenarios where clear answers are absent. To bridge this gap, we design the Agent Trading Arena, a virtual numerical game simulating complex economic systems through zero-sum games, where agents invest in stock portfolios. Our experiments reveal that LLMs, including GPT-4o, struggle with algebraic reasoning when dealing with plain-text stock data, often focusing on local details rather than global trends. In contrast, LLMs perform significantly better with geometric reasoning when presented with visual data, such as scatter plots or K-line charts, suggesting that visual representations enhance numerical reasoning. This capability is further improved by incorporating the reflection module, which aids in the analysis and interpretation of complex data. We validate our findings on NASDAQ Stock dataset, where LLMs demonstrate stronger reasoning with visual data compared to text. Our code and data are publicly available at this https URL. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在自然语言处理任务中的性能有了显著提升，但其在动态、未见过的任务中，特别是数值推理任务中的泛化能力仍存在挑战。现有基准主要评估LLMs在具有预定义最优解的问题上的表现，这可能与现实世界中没有明确答案的情景不一致。为了弥合这一差距，我们设计了Agent Trading Arena，这是一个通过零和博弈模拟复杂经济系统的虚拟数值游戏，在该游戏中，代理投资股票组合。我们的实验表明，当处理呈现文本形式的股票数据时，LLMs在代数推理方面挣扎，往往关注局部细节而非整体趋势。相比之下，当展示可视化数据（如散点图或K线图）时，LLMs在几何推理方面的表现显著更好，这表明可视化表示可以增强数值推理能力。通过引入反射模块，该能力进一步提升，有助于复杂数据的分析和解释。我们在NASDAQ股票数据集上验证了这些发现，显示LLMs在处理可视化数据方面表现出更强的推理能力。我们的代码和数据可在以下网址获取。 

---
# Language Models' Factuality Depends on the Language of Inquiry 

**Title (ZH)**: 语言模型的事实性取决于查询语言。 

**Authors**: Tushar Aggarwal, Kumar Tanmay, Ayush Agrawal, Kumar Ayush, Hamid Palangi, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17955)  

**Abstract**: Multilingual language models (LMs) are expected to recall factual knowledge consistently across languages, yet they often fail to transfer knowledge between languages even when they possess the correct information in one of the languages. For example, we find that an LM may correctly identify Rashed Al Shashai as being from Saudi Arabia when asked in Arabic, but consistently fails to do so when asked in English or Swahili. To systematically investigate this limitation, we introduce a benchmark of 10,000 country-related facts across 13 languages and propose three novel metrics: Factual Recall Score, Knowledge Transferability Score, and Cross-Lingual Factual Knowledge Transferability Score-to quantify factual recall and knowledge transferability in LMs across different languages. Our results reveal fundamental weaknesses in today's state-of-the-art LMs, particularly in cross-lingual generalization where models fail to transfer knowledge effectively across different languages, leading to inconsistent performance sensitive to the language used. Our findings emphasize the need for LMs to recognize language-specific factual reliability and leverage the most trustworthy information across languages. We release our benchmark and evaluation framework to drive future research in multilingual knowledge transfer. 

**Abstract (ZH)**: 多语言语言模型（LMs）在不同语言中一致地回忆事实知识存在期望，但它们往往在一种语言中有正确信息时仍无法在不同语言之间转移知识。为了系统地探讨这一限制，我们引入了一个包含13种语言中10,000条国家相关的事实的基准，并提出三项新的度量标准：事实回忆分值、知识可转移性分值和跨语言事实知识可转移性分值，以量化不同语言中LMs的事实回忆和知识可转移性。我们的结果揭示了当今最先进的LMs的基本不足之处，特别是在跨语言泛化方面，模型无法有效跨语言转移知识，导致使用不同语言时性能不一致。我们的发现强调LMs需要识别语言特定的事实可靠性，并在不同语言中利用最可信的信息。我们发布了该基准和评估框架，以推动多语言知识转移的未来研究。 

---
# Robust Polyp Detection and Diagnosis through Compositional Prompt-Guided Diffusion Models 

**Title (ZH)**: 通过组成部件提示导向扩散模型的结肠息肉鲁棒检测与诊断 

**Authors**: Jia Yu, Yan Zhu, Peiyao Fu, Tianyi Chen, Junbo Huang, Quanlin Li, Pinghong Zhou, Zhihua Wang, Fei Wu, Shuo Wang, Xian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17951)  

**Abstract**: Colorectal cancer (CRC) is a significant global health concern, and early detection through screening plays a critical role in reducing mortality. While deep learning models have shown promise in improving polyp detection, classification, and segmentation, their generalization across diverse clinical environments, particularly with out-of-distribution (OOD) data, remains a challenge. Multi-center datasets like PolypGen have been developed to address these issues, but their collection is costly and time-consuming. Traditional data augmentation techniques provide limited variability, failing to capture the complexity of medical images. Diffusion models have emerged as a promising solution for generating synthetic polyp images, but the image generation process in current models mainly relies on segmentation masks as the condition, limiting their ability to capture the full clinical context. To overcome these limitations, we propose a Progressive Spectrum Diffusion Model (PSDM) that integrates diverse clinical annotations-such as segmentation masks, bounding boxes, and colonoscopy reports-by transforming them into compositional prompts. These prompts are organized into coarse and fine components, allowing the model to capture both broad spatial structures and fine details, generating clinically accurate synthetic images. By augmenting training data with PSDM-generated samples, our model significantly improves polyp detection, classification, and segmentation. For instance, on the PolypGen dataset, PSDM increases the F1 score by 2.12% and the mean average precision by 3.09%, demonstrating superior performance in OOD scenarios and enhanced generalization. 

**Abstract (ZH)**: 结直肠癌（CRC）是全球重要的公共卫生问题，通过筛查进行早期检测在降低死亡率方面起着关键作用。尽管深度学习模型在提高息肉检测、分类和分割方面显示出潜力，但它们在多种临床环境下的泛化能力，尤其是面对分布外（OOD）数据时，仍面临挑战。为了应对这些问题，多中心数据集如PolypGen被开发出来，但其收集成本高且耗时。传统数据增强技术提供的可变性有限，无法捕捉医学图像的复杂性。扩散模型作为一种生成合成息肉图像的有前景解决方案已逐渐出现，但在当前模型中，图像生成过程主要依赖于分割掩码作为条件，限制了其捕捉完整临床上下文的能力。为克服这些限制，我们提出了一种渐进频谱扩散模型（PSDM），将多样化的临床注释，如分割掩码、边界框和结肠镜报告，转化为组成式提示，并组织为粗粒度和细粒度组件，使模型能够捕捉广泛的空间结构和细微细节，生成临床准确的合成图像。通过使用PSDM生成的样本增强训练数据，我们的模型显著提高了息肉检测、分类和分割的效果。例如，在PolypGen数据集中，PSDM使F1分数提高了2.12%，平均精确度提高了3.09%，表现出在OOD场景中的优越性能并增强了泛化能力。 

---
# DeepSeek-R1 Outperforms Gemini 2.0 Pro, OpenAI o1, and o3-mini in Bilingual Complex Ophthalmology Reasoning 

**Title (ZH)**: DeepSeek-R1 在双语复杂眼科推理任务中性能优于 Gemini 2.0 Pro、OpenAI o1 和 o3-mini 

**Authors**: Pusheng Xu, Yue Wu, Kai Jin, Xiaolan Chen, Mingguang He, Danli Shi  

**Link**: [PDF](https://arxiv.org/pdf/2502.17947)  

**Abstract**: Purpose: To evaluate the accuracy and reasoning ability of DeepSeek-R1 and three other recently released large language models (LLMs) in bilingual complex ophthalmology cases. Methods: A total of 130 multiple-choice questions (MCQs) related to diagnosis (n = 39) and management (n = 91) were collected from the Chinese ophthalmology senior professional title examination and categorized into six topics. These MCQs were translated into English using DeepSeek-R1. The responses of DeepSeek-R1, Gemini 2.0 Pro, OpenAI o1 and o3-mini were generated under default configurations between February 15 and February 20, 2025. Accuracy was calculated as the proportion of correctly answered questions, with omissions and extra answers considered incorrect. Reasoning ability was evaluated through analyzing reasoning logic and the causes of reasoning error. Results: DeepSeek-R1 demonstrated the highest overall accuracy, achieving 0.862 in Chinese MCQs and 0.808 in English MCQs. Gemini 2.0 Pro, OpenAI o1, and OpenAI o3-mini attained accuracies of 0.715, 0.685, and 0.692 in Chinese MCQs (all P<0.001 compared with DeepSeek-R1), and 0.746 (P=0.115), 0.723 (P=0.027), and 0.577 (P<0.001) in English MCQs, respectively. DeepSeek-R1 achieved the highest accuracy across five topics in both Chinese and English MCQs. It also excelled in management questions conducted in Chinese (all P<0.05). Reasoning ability analysis showed that the four LLMs shared similar reasoning logic. Ignoring key positive history, ignoring key positive signs, misinterpretation medical data, and too aggressive were the most common causes of reasoning errors. Conclusion: DeepSeek-R1 demonstrated superior performance in bilingual complex ophthalmology reasoning tasks than three other state-of-the-art LLMs. While its clinical applicability remains challenging, it shows promise for supporting diagnosis and clinical decision-making. 

**Abstract (ZH)**: 目的：评估DeepSeek-R1及其对比的三种最近发布的大型语言模型（LLMs）在双语复杂眼科病例中的准确性和推理能力。方法：收集了130道与诊断（n=39）和管理（n=91）相关的多项选择题（MCQs），并按照六个主题进行了分类。使用DeepSeek-R1将这些MCQs翻译成英文。在2025年2月15日至20日之间，四种LLMs（默认配置）生成了响应。准确性计算为正确回答的问题比例，未作答和额外答案视为错误。推理能力通过分析推理逻辑及其错误原因进行评估。结果：DeepSeek-R1在总体准确率上最高，中文MCQs为0.862，英文MCQs为0.808。Gemini 2.0 Pro、OpenAI o1和OpenAI o3-mini在中文MCQs中的准确率分别为0.715、0.685和0.692（所有值与DeepSeek-R1相比P<0.001），英文MCQs分别为0.746（P=0.115）、0.723（P=0.027）和0.577（P<0.001）。DeepSeek-R1在所有五个主题的中文和英文MCQs中的准确率最高。在中文管理问题中，其表现也最好（所有P<0.05）。推理能力分析显示，四种LLMs的推理逻辑相似。忽略关键阳性病史、忽略关键阳性体征、误解读医疗数据和过于激进是最常见的推理错误原因。结论：DeepSeek-R1在双语复杂眼科推理任务中表现优于三种其他最先进的LLMs。尽管其临床应用仍具有挑战性，但它在支持诊断和临床决策方面具有潜力。 

---
# Optimal Brain Apoptosis 

**Title (ZH)**: 最优大脑凋亡 

**Authors**: Mingyuan Sun, Zheng Fang, Jiaxu Wang, Junjie Jiang, Delei Kong, Chenming Hu, Yuetong Fang, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17941)  

**Abstract**: The increasing complexity and parameter count of Convolutional Neural Networks (CNNs) and Transformers pose challenges in terms of computational efficiency and resource demands. Pruning has been identified as an effective strategy to address these challenges by removing redundant elements such as neurons, channels, or connections, thereby enhancing computational efficiency without heavily compromising performance. This paper builds on the foundational work of Optimal Brain Damage (OBD) by advancing the methodology of parameter importance estimation using the Hessian matrix. Unlike previous approaches that rely on approximations, we introduce Optimal Brain Apoptosis (OBA), a novel pruning method that calculates the Hessian-vector product value directly for each parameter. By decomposing the Hessian matrix across network layers and identifying conditions under which inter-layer Hessian submatrices are non-zero, we propose a highly efficient technique for computing the second-order Taylor expansion of parameters. This approach allows for a more precise pruning process, particularly in the context of CNNs and Transformers, as validated in our experiments including VGG19, ResNet32, ResNet50, and ViT-B/16 on CIFAR10, CIFAR100 and Imagenet datasets. Our code is available at this https URL. 

**Abstract (ZH)**: 不断增长的卷积神经网络（CNNs）和变换器的复杂性和参数量给计算效率和资源需求带来了挑战。剪枝已被识别为一种有效的策略，通过移除冗余的元素如神经元、通道或连接，从而在不严重影响性能的情况下提升计算效率。本文在Optimal Brain Damage（OBD）的基础上，推进了使用Hessian矩阵估计参数重要性的方法。不同于以往依赖近似的方法，我们提出了Optimal Brain Apoptosis（OBA），一种新的剪枝方法，直接计算每个参数的Hessian-向量积值。通过在网络层间分解Hessian矩阵，并识别层间Hessian子矩阵非零的条件，我们提出了一种高效计算参数二阶泰勒展开的方法。这种方法在卷积神经网络（CNNs）和变换器（Transformers）中实现了更精确的剪枝过程，这一点在针对VGG19、ResNet32、ResNet50和ViT-B/16在CIFAR10、CIFAR100和ImageNet数据集上的实验中得到了验证。我们的代码可在以下链接获取。 

---
# Integrating Boosted learning with Differential Evolution (DE) Optimizer: A Prediction of Groundwater Quality Risk Assessment in Odisha 

**Title (ZH)**: 将增强学习与差分进化优化器相结合：奥dish地下水质量风险评估的预测 

**Authors**: Sonalika Subudhi, Alok Kumar Pati, Sephali Bose, Subhasmita Sahoo, Avipsa Pattanaik, Biswa Mohan Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2502.17929)  

**Abstract**: Groundwater is eventually undermined by human exercises, such as fast industrialization, urbanization, over-extraction, and contamination from agrarian and urban sources. From among the different contaminants, the presence of heavy metals like cadmium (Cd), chromium (Cr), arsenic (As), and lead (Pb) proves to have serious dangers when present in huge concentrations in groundwater. Long-term usage of these poisonous components may lead to neurological disorders, kidney failure and different sorts of cancer. To address these issues, this study developed a machine learning-based predictive model to evaluate the Groundwater Quality Index (GWQI) and identify the main contaminants which are affecting the water quality. It has been achieved with the help of a hybrid machine learning model i.e. LCBoost Fusion . The model has undergone several processes like data preprocessing, hyperparameter tuning using Differential Evolution (DE) optimization, and evaluation through cross-validation. The LCBoost Fusion model outperforms individual models (CatBoost and LightGBM), by achieving low RMSE (0.6829), MSE (0.5102), MAE (0.3147) and a high R$^2$ score of 0.9809. Feature importance analysis highlights Potassium (K), Fluoride (F) and Total Hardness (TH) as the most influential indicators of groundwater contamination. This research successfully demonstrates the application of machine learning in assessing groundwater quality risks in Odisha. The proposed LCBoost Fusion model offers a reliable and efficient approach for real-time groundwater monitoring and risk mitigation. These findings will help the environmental organizations and the policy makers to map out targeted places for sustainable groundwater management. Future work will focus on using remote sensing data and developing an interactive decision-making system for groundwater quality assessment. 

**Abstract (ZH)**: 地下水最终由于快速工业化、城市化、过度开采以及来自农业和城市来源的污染而遭到破坏。在这诸多污染物中，镉（Cd）、铬（Cr）、砷（As）和铅（Pb）等重金属在高浓度下对地下水造成了严重威胁。长期使用这些有毒成分可能会导致神经紊乱、肾衰竭和不同类型的癌症。为解决这些问题，本研究开发了一种基于机器学习的预测模型，以评估地下水质量指数（GWQI）并识别主要的污染物。该模型借助一种混合机器学习模型即LCBoost Fusion实现。经过数据预处理、使用差分进化（DE）优化的超参数调优以及交叉验证评估等多个过程，LCBoost Fusion模型优于单独使用的模型（CatBoost和LightGBM），其RMSE为0.6829，MSE为0.5102，MAE为0.3147，R$^2$得分为0.9809。特征重要性分析表明，钾（K）、氟（F）和总硬度（TH）是地下水污染最重要的指标。本研究成功展示了在奥里萨邦应用机器学习评估地下水质量风险的方法。提出的LCBoost Fusion模型提供了一种可靠且高效的实时地下水监控和风险缓解方法。这些发现将帮助环保组织和政策制定者确定可持续地下水管理的重点区域。未来工作将侧重于使用遥感数据并开发交互式决策系统来评估地下水质量。 

---
# Structure-prior Informed Diffusion Model for Graph Source Localization with Limited Data 

**Title (ZH)**: 基于结构先验的受限数据图源定位扩散模型 

**Authors**: Hongyi Chen, Jingtao Ding, Xiaojun Liang, Yong Li, Xiao-Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17928)  

**Abstract**: The source localization problem in graph information propagation is crucial for managing various network disruptions, from misinformation spread to infrastructure failures. While recent deep generative approaches have shown promise in this domain, their effectiveness is limited by the scarcity of real-world propagation data. This paper introduces SIDSL (\textbf{S}tructure-prior \textbf{I}nformed \textbf{D}iffusion model for \textbf{S}ource \textbf{L}ocalization), a novel framework that addresses three key challenges in limited-data scenarios: unknown propagation patterns, complex topology-propagation relationships, and class imbalance between source and non-source nodes. SIDSL incorporates topology-aware priors through graph label propagation and employs a propagation-enhanced conditional denoiser with a GNN-parameterized label propagation module (GNN-LP). Additionally, we propose a structure-prior biased denoising scheme that initializes from structure-based source estimations rather than random noise, effectively countering class imbalance issues. Experimental results across four real-world datasets demonstrate SIDSL's superior performance, achieving 7.5-13.3% improvements in F1 scores compared to state-of-the-art methods. Notably, when pretrained with simulation data of synthetic patterns, SIDSL maintains robust performance with only 10% of training data, surpassing baselines by more than 18.8%. These results highlight SIDSL's effectiveness in real-world applications where labeled data is scarce. 

**Abstract (ZH)**: 基于图信息传播的源定位问题对于管理和应对各种网络中断（从 misinformation 传播到基础设施故障）至关重要。尽管近期的深度生成方法在这一领域展现出了潜力，但它们的有效性受限于真实传播数据的稀缺。本文提出了SIDSL（结构先验导向的传播模型用于源定位），这是一种针对少量数据场景中三个关键挑战的新框架：未知的传播模式、复杂的拓扑-传播关系以及源节点与非源节点之间的类别不平衡。SIDSL 通过图标签传播引入了拓扑感知先验，并使用了传播增强条件降噪器，包含了一个基于 GNN 的标签传播模块（GNN-LP）。此外，我们提出了一种结构先验导向的降噪方案，从基于结构的源估测初始化，有效克服了类别不平衡问题。在四个真实世界数据集上的实验结果表明，SIDSL 在 F1 分数方面表现优异，比最先进的方法提高了 7.5-13.3%。特别地，预训练时使用合成模式的仿真数据，即使只有 10% 的训练数据，SIDSL 仍然表现出色，超过基线方法超过 18.8%。这些结果突显了 SIDSL 在缺乏标注数据的实际应用中的有效性。 

---
# Decoupled Graph Energy-based Model for Node Out-of-Distribution Detection on Heterophilic Graphs 

**Title (ZH)**: 异质图中节点离分布检测的解耦图能量模型 

**Authors**: Yuhan Chen, Yihong Luo, Yifan Song, Pengwen Dai, Jing Tang, Xiaochun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2502.17912)  

**Abstract**: Despite extensive research efforts focused on OOD detection on images, OOD detection on nodes in graph learning remains underexplored. The dependence among graph nodes hinders the trivial adaptation of existing approaches on images that assume inputs to be i.i.d. sampled, since many unique features and challenges specific to graphs are not considered, such as the heterophily issue. Recently, GNNSafe, which considers node dependence, adapted energy-based detection to the graph domain with state-of-the-art performance, however, it has two serious issues: 1) it derives node energy from classification logits without specifically tailored training for modeling data distribution, making it less effective at recognizing OOD data; 2) it highly relies on energy propagation, which is based on homophily assumption and will cause significant performance degradation on heterophilic graphs, where the node tends to have dissimilar distribution with its neighbors. To address the above issues, we suggest training EBMs by MLE to enhance data distribution modeling and remove energy propagation to overcome the heterophily issues. However, training EBMs via MLE requires performing MCMC sampling on both node feature and node neighbors, which is challenging due to the node interdependence and discrete graph topology. To tackle the sampling challenge, we introduce DeGEM, which decomposes the learning process into two parts: a graph encoder that leverages topology information for node representations and an energy head that operates in latent space. Extensive experiments validate that DeGEM, without OOD exposure during training, surpasses previous state-of-the-art methods, achieving an average AUROC improvement of 6.71% on homophilic graphs and 20.29% on heterophilic graphs, and even outperform methods trained with OOD exposure. Our code is available at: this https URL. 

**Abstract (ZH)**: 尽管在图像数据上的离域检测（OOD检测）已经得到了广泛的研究，但图学习中节点的离域检测仍缺乏探索。图节点之间的依赖关系阻碍了现有假设输入为独立同分布抽样图像方法的直接适应，因为许多与图特有的问题和挑战未被考虑，例如异质性问题。近期，GNNSafe 考虑到了节点依赖性，将基于能量的检测引入图域中，并取得了最先进的性能，然而它存在两个严重的问题：1）其节点能量来源于分类logits，没有专门针对建模数据分布的训练，使其在识别离域数据方面效果不佳；2）其高度依赖能量传播机制，该机制基于同质性假设，在异质图上会导致显著的性能下降，即节点与其邻居的分布差异较大。为解决上述问题，我们建议通过极大似然估计（MLE）训练能量分布模型以增强数据分布建模，并移除能量传播以克服异质性问题。然而，通过MLE训练能量分布模型需要在节点特征和节点邻居上进行马尔可夫链蒙特卡洛（MCMC）采样，这因节点间的依赖关系和离散的图拓扑结构而具有挑战性。为应对采样挑战，我们引入了DeGEM，它将学习过程分解为两部分：一个利用拓扑信息进行节点表示的图编码器和一个在潜在空间操作的能量头。大量实验验证了DeGEM在训练过程中无需暴露于离域数据的情况下，超过了之前的最先进的方法，在同质图上平均AUROC提高了6.71%，在异质图上提高了20.29%，甚至优于在暴露于离域数据下训练的方法。我们的代码可在：this https URL。 

---
# Enhancing Speech Quality through the Integration of BGRU and Transformer Architectures 

**Title (ZH)**: 通过结合BGRU和Transformer架构提升语音质量 

**Authors**: Souliman Alghnam, Mohammad Alhussien, Khaled Shaheen  

**Link**: [PDF](https://arxiv.org/pdf/2502.17911)  

**Abstract**: Speech enhancement plays an essential role in improving the quality of speech signals in noisy environments. This paper investigates the efficacy of integrating Bidirectional Gated Recurrent Units (BGRU) and Transformer models for speech enhancement tasks. Through a comprehensive experimental evaluation, our study demonstrates the superiority of this hybrid architecture over traditional methods and standalone models. The combined BGRU-Transformer framework excels in capturing temporal dependencies and learning complex signal patterns, leading to enhanced noise reduction and improved speech quality. Results show significant performance gains compared to existing approaches, highlighting the potential of this integrated model in real-world applications. The seamless integration of BGRU and Transformer architectures not only enhances system robustness but also opens the road for advanced speech processing techniques. This research contributes to the ongoing efforts in speech enhancement technology and sets a solid foundation for future investigations into optimizing model architectures, exploring many application scenarios, and advancing the field of speech processing in noisy environments. 

**Abstract (ZH)**: 演讲增强在噪声环境下的语音信号质量提升中发挥着关键作用。本文研究了将双向门控循环单元(BGRU)和Transformer模型集成应用于演讲增强任务的效果。通过全面的实验评估，我们的研究证明了这种混合架构优于传统方法和单一模型。结合的BGRU-Transformer框架在捕捉时序依赖性和学习复杂信号模式方面表现出色，从而提高了噪声抑制性能和语音质量。结果表明，与现有方法相比具有显著的性能提升，突显了该集成模型在实际应用中的潜力。BGRU和Transformer架构的无缝集成不仅增强了系统的鲁棒性，还为先进的语音处理技术铺平了道路。本文为语音增强技术的持续发展做出了贡献，并为未来优化模型架构、探索多种应用场景以及推进噪声环境下语音处理领域奠定了坚实的基础。 

---
# Scaling LLM Pre-training with Vocabulary Curriculum 

**Title (ZH)**: LLM预训练的词汇量递增 scaling LLM预训练与词汇 Curriculum 

**Authors**: Fangyuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17910)  

**Abstract**: Modern language models rely on static vocabularies, fixed before pretraining, in contrast to the adaptive vocabulary acquisition observed in human language learning. To bridge this gap, we introduce vocabulary curriculum learning, an approach that improves pretraining efficiency with log-linear scaling gains relative to vocabulary size. Our method alternates between entropy-guided vocabulary expansion and model optimization, enabling models to learn transferable representations across diverse tokenization granularities. This approach naturally gives rise to an optimal computation allocation pattern: longer tokens capture predictable content, while shorter tokens focus on more complex, harder-to-predict contexts. Experiments on small-scale GPT models demonstrate improved scaling efficiency, reinforcing the effectiveness of dynamic tokenization. We release our code to support further research and plan to extend our experiments to larger models and diverse domains. 

**Abstract (ZH)**: 现代语言模型依赖于预训练前固定的静态词汇表，而在人类语言学习中观察到的是适应性词汇获取。为弥合这一差距，我们提出了词汇课程学习，该方法通过相对于词汇量的对数线性缩放增益来提高预训练效率。该方法交替进行基于熵的词汇扩展和模型优化，使模型能够在不同的标记化粒度下学习可转移的表示。这种做法自然导致了最优的计算分配模式：较长的标记捕获可预测的内容，而较短的标记则专注于更复杂、更难以预测的上下文。在小型GPT模型上的实验显示了增强的缩放效率，证实了动态标记化的效果。我们发布了代码以支持进一步的研究，并计划将实验扩展到更大的模型和不同的领域。 

---
# FactFlow: Automatic Fact Sheet Generation and Customization from Tabular Dataset via AI Chain Design & Implementation 

**Title (ZH)**: FactFlow：通过AI链设计与实现从表格数据集自动生成和定制事实概要 

**Authors**: Minh Duc Vu, Jieshan Chen, Zhenchang Xing, Qinghua Lu, Xiwei Xu, Qian Fu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17909)  

**Abstract**: With the proliferation of data across various domains, there is a critical demand for tools that enable non-experts to derive meaningful insights without deep data analysis skills. To address this need, existing automatic fact sheet generation tools offer heuristic-based solutions to extract facts and generate stories. However, they inadequately grasp the semantics of data and struggle to generate narratives that fully capture the semantics of the dataset or align the fact sheet with specific user needs. Addressing these shortcomings, this paper introduces \tool, a novel tool designed for the automatic generation and customisation of fact sheets. \tool applies the concept of collaborative AI workers to transform raw tabular dataset into comprehensive, visually compelling fact sheets. We define effective taxonomy to profile AI worker for specialised tasks. Furthermore, \tool empowers users to refine these fact sheets through intuitive natural language commands, ensuring the final outputs align closely with individual preferences and requirements. Our user evaluation with 18 participants confirms that \tool not only surpasses state-of-the-art baselines in automated fact sheet production but also provides a positive user experience during customization tasks. 

**Abstract (ZH)**: 随着数据在各个领域中的泛滥，存在对无需深厚数据分析技能就能帮助非专家提取有意义见解的工具的迫切需求。为此，现有的自动事实表生成工具提供基于启发式的解决方案来提取事实并生成故事。然而，这些工具在理解数据语义方面存在不足，难以生成能够全面捕捉数据集语义或与特定用户需求相一致的叙述。为应对这些不足，本文介绍了\tool这一新型工具，旨在自动生成和定制事实表。\tool利用协作AI工作者的概念，将原始表格数据转换为全面且具有视觉吸引力的事实表。我们定义了有效的分类学来为专门任务配置AI工作者。此外，\tool通过直观的自然语言命令赋予用户进一步调整这些事实表的能力，确保最终输出与个人偏好和需求紧密契合。我们的用户评估（涉及18名参与者）表明，\tool不仅在自动事实表生成方面超过了现有的先进基准，还在定制任务中提供了积极的用户体验。 

---
# Knowledge-enhanced Multimodal ECG Representation Learning with Arbitrary-Lead Inputs 

**Title (ZH)**: 增强知识的多模态心电图表示学习：任意导联输入 

**Authors**: Che Liu, Cheng Ouyang, Zhongwei Wan, Haozhe Wang, Wenjia Bai, Rossella Arcucci  

**Link**: [PDF](https://arxiv.org/pdf/2502.17900)  

**Abstract**: Recent advances in multimodal ECG representation learning center on aligning ECG signals with paired free-text reports. However, suboptimal alignment persists due to the complexity of medical language and the reliance on a full 12-lead setup, which is often unavailable in under-resourced settings. To tackle these issues, we propose **K-MERL**, a knowledge-enhanced multimodal ECG representation learning framework. **K-MERL** leverages large language models to extract structured knowledge from free-text reports and employs a lead-aware ECG encoder with dynamic lead masking to accommodate arbitrary lead inputs. Evaluations on six external ECG datasets show that **K-MERL** achieves state-of-the-art performance in zero-shot classification and linear probing tasks, while delivering an average **16%** AUC improvement over existing methods in partial-lead zero-shot classification. 

**Abstract (ZH)**: 近年来，多模态心电图表示学习的进展集中在对齐心电图信号与配对的自由文本报告。但由于医疗语言的复杂性和对完整12导联设置的依赖，在资源不足的环境中，这种对齐仍然不尽如人意。为解决这些问题，我们提出了一种知识增强的多模态心电图表示学习框架 **K-MERL**。K-MERL 利用大型语言模型从自由文本报告中提取结构化知识，并采用Aware Leads的心电图编码器结合动态导联遮掩，以适应任意导联输入。在六个外部心电图数据集上的评估表明，K-MERL 在零样本分类和线性探测任务中达到了最先进的性能，同时在部分导联零样本分类中比现有方法平均提高了16%的AUC值。 

---
# VeriPlan: Integrating Formal Verification and LLMs into End-User Planning 

**Title (ZH)**: VeriPlan: 将形式验证与大规模语言模型集成到最终用户规划中 

**Authors**: Christine Lee, David Porfirio, Xinyu Jessica Wang, Kevin Zhao, Bilge Mutlu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17898)  

**Abstract**: Automated planning is traditionally the domain of experts, utilized in fields like manufacturing and healthcare with the aid of expert planning tools. Recent advancements in LLMs have made planning more accessible to everyday users due to their potential to assist users with complex planning tasks. However, LLMs face several application challenges within end-user planning, including consistency, accuracy, and user trust issues. This paper introduces VeriPlan, a system that applies formal verification techniques, specifically model checking, to enhance the reliability and flexibility of LLMs for end-user planning. In addition to the LLM planner, VeriPlan includes three additional core features -- a rule translator, flexibility sliders, and a model checker -- that engage users in the verification process. Through a user study (n=12), we evaluate VeriPlan, demonstrating improvements in the perceived quality, usability, and user satisfaction of LLMs. Our work shows the effective integration of formal verification and user-control features with LLMs for end-user planning tasks. 

**Abstract (ZH)**: 自动规划传统上是专家的领域，借助专家规划工具在制造和医疗等领域中应用。近期大型语言模型的进步使规划任务因能够辅助用户处理复杂规划任务而更加适用于普通用户。然而，大型语言模型在面向最终用户的规划中仍面临一致性、准确性和用户信任等问题。本文介绍了一种名为VeriPlan的系统，该系统利用形式验证技术——模型检查——来增强大型语言模型在最终用户规划中的可靠性和灵活性。VeriPlan还包括三个核心功能——规则翻译器、灵活性调节器和模型检查器——以使用户参与到验证过程。通过一项用户研究（n=12），我们评估了VeriPlan，展示了大型语言模型在感知质量、易用性和用户满意度方面的改进。我们的工作展示了将形式验证和用户控制功能有效集成到大型语言模型中以完成面向最终用户的规划任务的有效性。 

---
# Sample-efficient diffusion-based control of complex nonlinear systems 

**Title (ZH)**: 基于扩散的复杂非线性系统高效样本控制 

**Authors**: Hongyi Chen, Jingtao Ding, Jianhai Shu, Xinchun Yu, Xiaojun Liang, Yong Li, Xiao-Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17893)  

**Abstract**: Complex nonlinear system control faces challenges in achieving sample-efficient, reliable performance. While diffusion-based methods have demonstrated advantages over classical and reinforcement learning approaches in long-term control performance, they are limited by sample efficiency. This paper presents SEDC (Sample-Efficient Diffusion-based Control), a novel diffusion-based control framework addressing three core challenges: high-dimensional state-action spaces, nonlinear system dynamics, and the gap between non-optimal training data and near-optimal control solutions. Through three innovations - Decoupled State Diffusion, Dual-Mode Decomposition, and Guided Self-finetuning - SEDC achieves 39.5\%-49.4\% better control accuracy than baselines while using only 10\% of the training samples, as validated across three complex nonlinear dynamic systems. Our approach represents a significant advancement in sample-efficient control of complex nonlinear systems. The implementation of the code can be found at this https URL. 

**Abstract (ZH)**: 基于扩散的方法在实现复杂非线性系统高效可靠控制方面面临挑战。虽然基于扩散的方法在长期控制性能上优于传统和强化学习方法，但它们受限于样本效率。本文提出了SEDC（高效基于扩散的控制），一种解决三个核心挑战的新颖基于扩散的控制框架：高维状态-动作空间、非线性系统动力学以及非最优训练数据与近最优控制解之间的差距。通过三项创新——解耦状态扩散、双模式分解和引导自微调，SEDC在三个复杂非线性动态系统验证中仅使用10%的训练样本就实现了39.5%-49.4%的控制精度改进。我们的方法代表了复杂非线性系统高效控制的重要进步。相关代码实现可访问该网址。 

---
# Arrhythmia Classification from 12-Lead ECG Signals Using Convolutional and Transformer-Based Deep Learning Models 

**Title (ZH)**: 基于卷积和变换器的深度学习模型的12导联心电图心律失常分类 

**Authors**: Andrei Apostol, Maria Nutu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17887)  

**Abstract**: In Romania, cardiovascular problems are the leading cause of death, accounting for nearly one-third of annual fatalities. The severity of this situation calls for innovative diagnosis method for cardiovascular diseases. This article aims to explore efficient, light-weight and rapid methods for arrhythmia diagnosis, in resource-constrained healthcare settings. Due to the lack of Romanian public medical data, we trained our systems using international public datasets, having in mind that the ECG signals are the same regardless the patients' nationality. Within this purpose, we combined multiple datasets, usually used in the field of arrhythmias classification: PTB-XL electrocardiography dataset , PTB Diagnostic ECG Database, China 12-Lead ECG Challenge Database, Georgia 12-Lead ECG Challenge Database, and St. Petersburg INCART 12-lead Arrhythmia Database. For the input data, we employed ECG signal processing methods, specifically a variant of the Pan-Tompkins algorithm, useful in arrhythmia classification because it provides a robust and efficient method for detecting QRS complexes in ECG signals. Additionally, we used machine learning techniques, widely used for the task of classification, including convolutional neural networks (1D CNNs, 2D CNNs, ResNet) and Vision Transformers (ViTs). The systems were evaluated in terms of accuracy and F1 score. We annalysed our dataset from two perspectives. First, we fed the systems with the ECG signals and the GRU-based 1D CNN model achieved the highest accuracy of 93.4% among all the tested architectures. Secondly, we transformed ECG signals into images and the CNN2D model achieved an accuracy of 92.16%. 

**Abstract (ZH)**: 在罗马尼亚，心血管问题是最主要的死亡原因，约占年度死亡人数的三分之一。为了应对这一严峻形势，本文旨在探索资源受限 healthcare 环境下高效、轻量化和快速的心律失常诊断方法。由于缺乏罗马尼亚公共医疗数据，我们使用国际公开数据集进行系统训练，考虑到心电图信号与患者国籍无关。为此，我们将心律失常分类领域常用的多个数据集结合起来，包括 PTB-XL 心电图数据集、PTB 诊断心电图数据库、中国12导联心电图挑战数据库、格鲁吉亚12导联心电图挑战数据库以及圣彼得堡 INCART 12导联心律失常数据库。对于输入数据，我们采用心电图信号处理方法，特别是改进的 Pan-Tompkins 算法，因为该算法能提供一种稳健且高效的方法来检测心电图信号中的 QRS 波群。此外，我们还采用了卷积神经网络（1D CNN、2D CNN、ResNet）和视觉变换器（ViTs）等机器学习技术，用于分类任务。系统从准确率和F1分数两个方面进行了评估。我们从两个角度分析了数据集。首先，我们用心电图信号喂给系统，GRU 基于的 1D CNN 模型取得了所有测试架构中最高的准确率 93.4%。其次，我们将心电图信号转换为图像，2D CNN 模型的准确率为 92.16%。 

---
# A graph neural network-based multispectral-view learning model for diabetic macular ischemia detection from color fundus photographs 

**Title (ZH)**: 基于图神经网络的多光谱视网膜图像糖尿病黄斑缺血检测模型 

**Authors**: Qinghua He, Hongyang Jiang, Danqi Fang, Dawei Yang, Truong X. Nguyen, Anran Ran, Clement C. Tham, Simon K. H. Szeto, Sobha Sivaprasad, Carol Y. Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2502.17886)  

**Abstract**: Diabetic macular ischemia (DMI), marked by the loss of retinal capillaries in the macular area, contributes to vision impairment in patients with diabetes. Although color fundus photographs (CFPs), combined with artificial intelligence (AI), have been extensively applied in detecting various eye diseases, including diabetic retinopathy (DR), their applications in detecting DMI remain unexplored, partly due to skepticism among ophthalmologists regarding its feasibility. In this study, we propose a graph neural network-based multispectral view learning (GNN-MSVL) model designed to detect DMI from CFPs. The model leverages higher spectral resolution to capture subtle changes in fundus reflectance caused by ischemic tissue, enhancing sensitivity to DMI-related features. The proposed approach begins with computational multispectral imaging (CMI) to reconstruct 24-wavelength multispectral fundus images from CFPs. ResNeXt101 is employed as the backbone for multi-view learning to extract features from the reconstructed images. Additionally, a GNN with a customized jumper connection strategy is designed to enhance cross-spectral relationships, facilitating comprehensive and efficient multispectral view learning. The study included a total of 1,078 macula-centered CFPs from 1,078 eyes of 592 patients with diabetes, of which 530 CFPs from 530 eyes of 300 patients were diagnosed with DMI. The model achieved an accuracy of 84.7 percent and an area under the receiver operating characteristic curve (AUROC) of 0.900 (95 percent CI: 0.852-0.937) on eye-level, outperforming both the baseline model trained from CFPs and human experts (p-values less than 0.01). These findings suggest that AI-based CFP analysis holds promise for detecting DMI, contributing to its early and low-cost screening. 

**Abstract (ZH)**: 糖尿病黄斑缺血（DMI）的图神经网络多光谱视图学习模型 

---
# From underwater to aerial: a novel multi-scale knowledge distillation approach for coral reef monitoring 

**Title (ZH)**: 从水下到空中：一种新型多尺度知识蒸馏方法用于珊瑚礁监测 

**Authors**: Matteo Contini, Victor Illien, Julien Barde, Sylvain Poulain, Serge Bernard, Alexis Joly, Sylvain Bonhommeau  

**Link**: [PDF](https://arxiv.org/pdf/2502.17883)  

**Abstract**: Drone-based remote sensing combined with AI-driven methodologies has shown great potential for accurate mapping and monitoring of coral reef ecosystems. This study presents a novel multi-scale approach to coral reef monitoring, integrating fine-scale underwater imagery with medium-scale aerial imagery. Underwater images are captured using an Autonomous Surface Vehicle (ASV), while aerial images are acquired with an aerial drone. A transformer-based deep-learning model is trained on underwater images to detect the presence of 31 classes covering various coral morphotypes, associated fauna, and habitats. These predictions serve as annotations for training a second model applied to aerial images. The transfer of information across scales is achieved through a weighted footprint method that accounts for partial overlaps between underwater image footprints and aerial image tiles. The results show that the multi-scale methodology successfully extends fine-scale classification to larger reef areas, achieving a high degree of accuracy in predicting coral morphotypes and associated habitats. The method showed a strong alignment between underwater-derived annotations and ground truth data, reflected by an AUC (Area Under the Curve) score of 0.9251. This shows that the integration of underwater and aerial imagery, supported by deep-learning models, can facilitate scalable and accurate reef assessments. This study demonstrates the potential of combining multi-scale imaging and AI to facilitate the monitoring and conservation of coral reefs. Our approach leverages the strengths of underwater and aerial imagery, ensuring the precision of fine-scale analysis while extending it to cover a broader reef area. 

**Abstract (ZH)**: 基于无人机的遥感结合AI驱动方法在珊瑚礁生态系统的精确制图和监测中展现出巨大潜力。本研究提出了一种新的多尺度珊瑚礁监测方法，将精细尺度的水下影像与中尺度的航空影像整合。水下影像使用自主水面车辆（ASV）获取，而航空影像通过无人机获得。采用基于变换器的深度学习模型在水下影像上训练，以检测31类各种珊瑚形态、相关生物及栖息地的存在。这些预测作为注释用于训练应用于航空影像的第二个模型。通过加权足迹方法实现跨尺度信息传递，该方法考虑了水下影像足迹与航空影像瓦片之间的部分重叠。结果显示，多尺度方法成功地将精细尺度分类扩展到更大的珊瑚礁区域，预测珊瑚形态和相关栖息地的准确性非常高。该方法在水下衍生注释与地面真实数据之间显示出强烈的对齐，AUC（曲线下面积）得分为0.9251。这表明结合水下和航空影像并通过深度学习模型进行集成，可以促进珊瑚礁评估的可扩展性和准确性。本研究展示了将多尺度成像与AI结合以促进珊瑚礁监测和保护的潜力。我们的方法充分利用了水下和航空影像的优点，在保持精细尺度分析精度的同时将其扩展到覆盖更广阔的珊瑚礁区域。 

---
# Contrastive Learning with Nasty Noise 

**Title (ZH)**: 带恶劣噪声的对比学习 

**Authors**: Ziruo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.17872)  

**Abstract**: Contrastive learning has emerged as a powerful paradigm for self-supervised representation learning. This work analyzes the theoretical limits of contrastive learning under nasty noise, where an adversary modifies or replaces training samples. Using PAC learning and VC-dimension analysis, lower and upper bounds on sample complexity in adversarial settings are established. Additionally, data-dependent sample complexity bounds based on the l2-distance function are derived. 

**Abstract (ZH)**: 对比学习在恶劣噪声下的理论极限：对抗设置下的样本复杂性分析 

---
# ASurvey: Spatiotemporal Consistency in Video Generation 

**Title (ZH)**: 时空一致性在视频生成中的调查 

**Authors**: Zhiyu Yin, Kehai Chen, Xuefeng Bai, Ruili Jiang, Juntao Li, Hongdong Li, Jin Liu, Yang Xiang, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17863)  

**Abstract**: Video generation, by leveraging a dynamic visual generation method, pushes the boundaries of Artificial Intelligence Generated Content (AIGC). Video generation presents unique challenges beyond static image generation, requiring both high-quality individual frames and temporal coherence to maintain consistency across the spatiotemporal sequence. Recent works have aimed at addressing the spatiotemporal consistency issue in video generation, while few literature review has been organized from this perspective. This gap hinders a deeper understanding of the underlying mechanisms for high-quality video generation. In this survey, we systematically review the recent advances in video generation, covering five key aspects: foundation models, information representations, generation schemes, post-processing techniques, and evaluation metrics. We particularly focus on their contributions to maintaining spatiotemporal consistency. Finally, we discuss the future directions and challenges in this field, hoping to inspire further efforts to advance the development of video generation. 

**Abstract (ZH)**: 通过利用动态视觉生成方法，视频生成推动了人工智能生成内容（AIGC）的边界。视频生成超越了静态图像生成，提出了新的挑战，需要高质量的单帧和时间一致性来维持时空序列的一致性。尽管近期的研究致力于解决视频生成中时空一致性的问题，但缺乏从这一视角组织的文献综述。这阻碍了对高质量视频生成背后机制的深入理解。在本文综述中，我们系统地回顾了视频生成的 Recent Advances，涵盖了五方面关键内容：基础模型、信息表示、生成方案、后处理技术和评估指标。特别关注这些方面对维持时空一致性的贡献。最后，我们讨论了该领域的未来方向和挑战，希望激发进一步的努力以推动视频生成的发展。 

---
# Say Less, Mean More: Leveraging Pragmatics in Retrieval-Augmented Generation 

**Title (ZH)**: 说的更少，意味更多：利用语用学在检索增强生成中的作用 

**Authors**: Haris Riaz, Ellen Riloff, Mihai Surdeanu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17839)  

**Abstract**: We propose a simple, unsupervised method that injects pragmatic principles in retrieval-augmented generation (RAG) frameworks such as Dense Passage Retrieval~\cite{karpukhin2020densepassageretrievalopendomain} to enhance the utility of retrieved contexts. Our approach first identifies which sentences in a pool of documents retrieved by RAG are most relevant to the question at hand, cover all the topics addressed in the input question and no more, and then highlights these sentences within their context, before they are provided to the LLM, without truncating or altering the context in any other way. We show that this simple idea brings consistent improvements in experiments on three question answering tasks (ARC-Challenge, PubHealth and PopQA) using five different LLMs. It notably enhances relative accuracy by up to 19.7\% on PubHealth and 10\% on ARC-Challenge compared to a conventional RAG system. 

**Abstract (ZH)**: 我们提出一种简单的无监督方法，将语用原则注入到检索增强生成（RAG）框架中，如密集段落检索（DPR）~\cite{karpukhin2020densepassageretrievalopendomain}，以增强检索上下文的实用性。该方法首先识别RAG检索出的文档集中哪些句子与问题最相关，覆盖输入问题涉及的所有话题但不过多包含无关内容，然后在将这些句子提供给LLM之前突出显示这些句子，而不会以任何方式截断或修改上下文。实验结果显示，该简单想法在三个问答任务（ARC-Challenge、PubHealth和PopQA）上使用五种不同LLM时均带来了持续改进。特别是在PubHealth和ARC-Challenge任务上，相对于传统RAG系统，相对准确性分别提升19.7%和10%。 

---
# MM-PoisonRAG: Disrupting Multimodal RAG with Local and Global Poisoning Attacks 

**Title (ZH)**: MM-PoisonRAG: 阻断多模态RAG的局部和全局中毒攻击 

**Authors**: Hyeonjeong Ha, Qiusi Zhan, Jeonghwan Kim, Dimitrios Bralios, Saikrishna Sanniboina, Nanyun Peng, Kai-wei Chang, Daniel Kang, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2502.17832)  

**Abstract**: Multimodal large language models (MLLMs) equipped with Retrieval Augmented Generation (RAG) leverage both their rich parametric knowledge and the dynamic, external knowledge to excel in tasks such as Question Answering. While RAG enhances MLLMs by grounding responses in query-relevant external knowledge, this reliance poses a critical yet underexplored safety risk: knowledge poisoning attacks, where misinformation or irrelevant knowledge is intentionally injected into external knowledge bases to manipulate model outputs to be incorrect and even harmful. To expose such vulnerabilities in multimodal RAG, we propose MM-PoisonRAG, a novel knowledge poisoning attack framework with two attack strategies: Localized Poisoning Attack (LPA), which injects query-specific misinformation in both text and images for targeted manipulation, and Globalized Poisoning Attack (GPA) to provide false guidance during MLLM generation to elicit nonsensical responses across all queries. We evaluate our attacks across multiple tasks, models, and access settings, demonstrating that LPA successfully manipulates the MLLM to generate attacker-controlled answers, with a success rate of up to 56% on MultiModalQA. Moreover, GPA completely disrupts model generation to 0% accuracy with just a single irrelevant knowledge injection. Our results highlight the urgent need for robust defenses against knowledge poisoning to safeguard multimodal RAG frameworks. 

**Abstract (ZH)**: 多模态大语言模型的知识中毒攻击：MM-PoisonRAG框架 

---
# CAML: Collaborative Auxiliary Modality Learning for Multi-Agent Systems 

**Title (ZH)**: CAML：多代理系统中的协作辅助模态学习 

**Authors**: Rui Liu, Yu Shen, Peng Gao, Pratap Tokekar, Ming Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.17821)  

**Abstract**: Multi-modality learning has become a crucial technique for improving the performance of machine learning applications across domains such as autonomous driving, robotics, and perception systems. While existing frameworks such as Auxiliary Modality Learning (AML) effectively utilize multiple data sources during training and enable inference with reduced modalities, they primarily operate in a single-agent context. This limitation is particularly critical in dynamic environments, such as connected autonomous vehicles (CAV), where incomplete data coverage can lead to decision-making blind spots. To address these challenges, we propose Collaborative Auxiliary Modality Learning ($\textbf{CAML}$), a novel multi-agent multi-modality framework that enables agents to collaborate and share multimodal data during training while allowing inference with reduced modalities per agent during testing. We systematically analyze the effectiveness of $\textbf{CAML}$ from the perspective of uncertainty reduction and data coverage, providing theoretical insights into its advantages over AML. Experimental results in collaborative decision-making for CAV in accident-prone scenarios demonstrate that \ours~achieves up to a ${\bf 58.13}\%$ improvement in accident detection. Additionally, we validate $\textbf{CAML}$ on real-world aerial-ground robot data for collaborative semantic segmentation, achieving up to a ${\bf 10.61}\%$ improvement in mIoU. 

**Abstract (ZH)**: 多模态学习已成为提高自主驾驶、机器人技术和感知系统等领域机器学习应用性能的关键技术。虽然现有的框架如辅助模态学习（AML）在训练期间有效利用多种数据源并在推理时减少模态数量方面表现出色，但它们主要在这种代理单一的背景下运作。这一限制在诸如连接的自动驾驶车辆（CAV）等动态环境中尤为重要，不完整的数据覆盖可能导致决策盲点。为应对这些挑战，我们提出了协作辅助模态学习（CAML），这是一种新型多代理多模态框架，使代理在训练期间能够协作和共享多模态数据，并在测试时每个代理使用减少的模态进行推理。我们从不确定性减少和数据覆盖的角度系统分析了CAML的有效性，提供了与AML相比其优势的理论见解。在事故多发场景中的协作决策实验结果显示，CAML在事故检测方面可实现高达58.13%的提升。此外，我们还在真实的空地机器人数据上验证了CAML在协作语义分割中的应用，mIoU提升了高达10.61%。 

---
# An Overview of Large Language Models for Statisticians 

**Title (ZH)**: 大型语言模型Overview for Statisticians 

**Authors**: Wenlong Ji, Weizhe Yuan, Emily Getzen, Kyunghyun Cho, Michael I. Jordan, Song Mei, Jason E Weston, Weijie J. Su, Jing Xu, Linjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17814)  

**Abstract**: Large Language Models (LLMs) have emerged as transformative tools in artificial intelligence (AI), exhibiting remarkable capabilities across diverse tasks such as text generation, reasoning, and decision-making. While their success has primarily been driven by advances in computational power and deep learning architectures, emerging problems -- in areas such as uncertainty quantification, decision-making, causal inference, and distribution shift -- require a deeper engagement with the field of statistics. This paper explores potential areas where statisticians can make important contributions to the development of LLMs, particularly those that aim to engender trustworthiness and transparency for human users. Thus, we focus on issues such as uncertainty quantification, interpretability, fairness, privacy, watermarking and model adaptation. We also consider possible roles for LLMs in statistical analysis. By bridging AI and statistics, we aim to foster a deeper collaboration that advances both the theoretical foundations and practical applications of LLMs, ultimately shaping their role in addressing complex societal challenges. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已成为人工智能（AI）领域的变革性工具，展示出在文本生成、推理和决策等多种任务中的卓越能力。虽然其成功主要得益于计算能力的提升和深度学习架构的进步，但在不确定性量化、决策制定、因果推断和分布迁移等问题领域出现的新挑战，则需要统计学领域更深入的参与。本文探讨了统计学家如何在促进LLMs的发展中发挥重要作用，特别是那些旨在增强人类用户信任和透明度的努力。因此，我们重点关注不确定性量化、可解释性、公平性、隐私、水印和模型适应等议题。我们还考虑了LLMs在统计分析中的潜在角色。通过弥合AI与统计学之间的差距，我们旨在促进更深层次的协作，推动LLMs理论基础和实际应用的进步，最终塑造其在解决复杂社会挑战中的角色。 

---
# Research on Enhancing Cloud Computing Network Security using Artificial Intelligence Algorithms 

**Title (ZH)**: 基于人工 Intelligence 算法提升云 computing 网络安全性研究 

**Authors**: Yuqing Wang, Xiao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17801)  

**Abstract**: Cloud computing environments are increasingly vulnerable to security threats such as distributed denial-of-service (DDoS) attacks and SQL injection. Traditional security mechanisms, based on rule matching and feature recognition, struggle to adapt to evolving attack strategies. This paper proposes an adaptive security protection framework leveraging deep learning to construct a multi-layered defense architecture. The proposed system is evaluated in a real-world business environment, achieving a detection accuracy of 97.3%, an average response time of 18 ms, and an availability rate of 99.999%. Experimental results demonstrate that the proposed method significantly enhances detection accuracy, response efficiency, and resource utilization, offering a novel and effective approach to cloud computing security. 

**Abstract (ZH)**: 基于深度学习的适应性安全保护框架在云计算安全中的应用研究 

---
# Synthia: Novel Concept Design with Affordance Composition 

**Title (ZH)**: Synthia: 基于功能组合的新概念设计 

**Authors**: Xiaomeng Jin, Hyeonjeong Ha, Jeonghwan Kim, Jiateng Liu, Zhenhailong Wang, Khanh Duy Nguyen, Ansel Blume, Nanyun Peng, Kai-wei Chang, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2502.17793)  

**Abstract**: Text-to-image (T2I) models enable rapid concept design, making them widely used in AI-driven design. While recent studies focus on generating semantic and stylistic variations of given design concepts, functional coherence--the integration of multiple affordances into a single coherent concept--remains largely overlooked. In this paper, we introduce SYNTHIA, a framework for generating novel, functionally coherent designs based on desired affordances. Our approach leverages a hierarchical concept ontology that decomposes concepts into parts and affordances, serving as a crucial building block for functionally coherent design. We also develop a curriculum learning scheme based on our ontology that contrastively fine-tunes T2I models to progressively learn affordance composition while maintaining visual novelty. To elaborate, we (i) gradually increase affordance distance, guiding models from basic concept-affordance association to complex affordance compositions that integrate parts of distinct affordances into a single, coherent form, and (ii) enforce visual novelty by employing contrastive objectives to push learned representations away from existing concepts. Experimental results show that SYNTHIA outperforms state-of-the-art T2I models, demonstrating absolute gains of 25.1% and 14.7% for novelty and functional coherence in human evaluation, respectively. 

**Abstract (ZH)**: 基于期望功能的新型功能连贯设计生成框架SYNTHIA 

---
# AIR: Complex Instruction Generation via Automatic Iterative Refinement 

**Title (ZH)**: AIR：通过自动迭代 refining 的复杂指令生成 

**Authors**: Wei Liu, Yancheng He, Hui Huang, Chengwei Hu, Jiaheng Liu, Shilong Li, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.17787)  

**Abstract**: With the development of large language models, their ability to follow simple instructions has significantly improved. However, adhering to complex instructions remains a major challenge. Current approaches to generating complex instructions are often irrelevant to the current instruction requirements or suffer from limited scalability and diversity. Moreover, methods such as back-translation, while effective for simple instruction generation, fail to leverage the rich contents and structures in large web corpora. In this paper, we propose a novel automatic iterative refinement framework to generate complex instructions with constraints, which not only better reflects the requirements of real scenarios but also significantly enhances LLMs' ability to follow complex instructions. The AIR framework consists of two stages: (1)Generate an initial instruction from a document; (2)Iteratively refine instructions with LLM-as-judge guidance by comparing the model's output with the document to incorporate valuable constraints. Finally, we construct the AIR-10K dataset with 10K complex instructions and demonstrate that instructions generated with our approach significantly improve the model's ability to follow complex instructions, outperforming existing methods for instruction generation. 

**Abstract (ZH)**: 随着大型语言模型的发展，它们遵循简单指令的能力显著提高，但遵守复杂指令仍然是一项重大挑战。当前生成复杂指令的方法往往与当前指令要求无关，或者受限于有限的可扩展性和多样性。此外，虽然反向翻译等方法对简单指令生成有效，但无法充分利用大规模网页语料库中的丰富内容和结构。在本文中，我们提出了一种新颖的自动迭代 refinement 框架，以生成受约束的复杂指令，不仅能更好地反映实际场景的需求，还能显著增强大型语言模型遵循复杂指令的能力。该 AIR 框架包含两个阶段：（1）从文档生成初始指令；（2）通过将模型输出与文档进行比较，利用 AIR 判官的指导进行迭代指令 refinement，逐步纳入有价值的需求约束。最后，我们构建了包含10,000条复杂指令的AIR-10K数据集，并证明使用我们方法生成的指令显著提高了模型遵循复杂指令的能力，优于现有指令生成方法。 

---
# Uncertainty Quantification for LLM-Based Survey Simulations 

**Title (ZH)**: 基于LLM的调查模拟中的不确定性量化 

**Authors**: Chengpiao Huang, Yuhang Wu, Kaizheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17773)  

**Abstract**: We investigate the reliable use of simulated survey responses from large language models (LLMs) through the lens of uncertainty quantification. Our approach converts synthetic data into confidence sets for population parameters of human responses, addressing the distribution shift between the simulated and real populations. A key innovation lies in determining the optimal number of simulated responses: too many produce overly narrow confidence sets with poor coverage, while too few yield excessively loose estimates. To resolve this, our method adaptively selects the simulation sample size, ensuring valid average-case coverage guarantees. It is broadly applicable to any LLM, irrespective of its fidelity, and any procedure for constructing confidence sets. Additionally, the selected sample size quantifies the degree of misalignment between the LLM and the target human population. We illustrate our method on real datasets and LLMs. 

**Abstract (ZH)**: 我们通过不确定性量化的角度调查了大规模语言模型（LLMs）模拟调查响应的可靠使用方法。我们的方法将合成数据转换为人类响应总体参数的置信集，解决了模拟群体与真实群体之间的分布偏移问题。这一创新的关键在于确定模拟响应的最佳数量：数量过多会产生覆盖不佳的过窄置信区间，而数量过少则会导致过于宽松的估计。为解决这一问题，我们的方法自适应地选择模拟样本大小，确保有效的平均情况覆盖率保证。该方法适用于任何LLM，无论其保真度如何，以及任何置信区间构建方法。此外，选定的样本大小量化了LLM与目标人类群体之间的不一致程度。我们在实际数据集和LLM上展示了该方法。 

---
# Sample Selection via Contrastive Fragmentation for Noisy Label Regression 

**Title (ZH)**: 基于对比碎片化的选择性采样用于噪声标签回归 

**Authors**: Chris Dongjoo Kim, Sangwoo Moon, Jihwan Moon, Dongyeon Woo, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.17771)  

**Abstract**: As with many other problems, real-world regression is plagued by the presence of noisy labels, an inevitable issue that demands our attention. Fortunately, much real-world data often exhibits an intrinsic property of continuously ordered correlations between labels and features, where data points with similar labels are also represented with closely related features. In response, we propose a novel approach named ConFrag, where we collectively model the regression data by transforming them into disjoint yet contrasting fragmentation pairs. This enables the training of more distinctive representations, enhancing the ability to select clean samples. Our ConFrag framework leverages a mixture of neighboring fragments to discern noisy labels through neighborhood agreement among expert feature extractors. We extensively perform experiments on six newly curated benchmark datasets of diverse domains, including age prediction, price prediction, and music production year estimation. We also introduce a metric called Error Residual Ratio (ERR) to better account for varying degrees of label noise. Our approach consistently outperforms fourteen state-of-the-art baselines, being robust against symmetric and random Gaussian label noise. 

**Abstract (ZH)**: 就像许多其他问题一样，实际世界的回归问题受到噪声标签的困扰，这是一个不可避免的问题，需要我们关注。幸运的是，大量实际数据往往表现出标签和特征之间内在的连续有序关联性，其中具有相似标签的数据点也表现为密切相关的特征。为应对这一挑战，我们提出了一种名为ConFrag的新方法，通过将回归数据转换为既不相连又具有对比性的碎片对来进行集体建模。这使得能够训练出更加独特的表示，增强选择清洁样本的能力。我们的ConFrag框架利用邻近碎片的混合，通过专家特征提取器之间的邻域一致性来区分噪声标签。我们在六个新编curated的不同领域基准数据集上进行了广泛实验，包括年龄预测、价格预测和音乐制作年份估计。我们还引入了一个称为错误残差比率（ERR）的指标，以更好地反映不同水平的标签噪声。我们的方法在对抗对称和随机高斯噪声的标注数据方面表现出色，始终优于十四种最先进的基线方法。 

---
# DeepSeek vs. ChatGPT: A Comparative Study for Scientific Computing and Scientific Machine Learning Tasks 

**Title (ZH)**: DeepSeek vs. ChatGPT: 科学计算和科学机器学习任务的比较研究 

**Authors**: Qile Jiang, Zhiwei Gao, George Em Karniadakis  

**Link**: [PDF](https://arxiv.org/pdf/2502.17764)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for tackling a wide range of problems, including those in scientific computing, particularly in solving partial differential equations (PDEs). However, different models exhibit distinct strengths and preferences, resulting in varying levels of performance. In this paper, we compare the capabilities of the most advanced LLMs--ChatGPT and DeepSeek--along with their reasoning-optimized versions in addressing computational challenges. Specifically, we evaluate their proficiency in solving traditional numerical problems in scientific computing as well as leveraging scientific machine learning techniques for PDE-based problems. We designed all our experiments so that a non-trivial decision is required, e.g. defining the proper space of input functions for neural operator learning. Our findings reveal that the latest model, ChatGPT o3-mini-high, usually delivers the most accurate results while also responding significantly faster than its reasoning counterpart, DeepSeek R1. This enhanced speed and accuracy make ChatGPT o3-mini-high a more practical and efficient choice for diverse computational tasks at this juncture. 

**Abstract (ZH)**: 大型语言模型（LLMs）在科学计算中的应用日益增多，特别是用于求解偏微分方程（PDEs）问题。然而，不同模型表现出不同的优势和偏好，导致性能差异。本文比较了最先进的LLMs——ChatGPT和DeepSeek及其推理优化版本在处理计算挑战方面的能力。具体来说，我们评估了它们在解决科学计算中的传统数值问题以及利用科学机器学习技术求解基于PDE的问题方面的 proficiency。我们设计的所有实验都要求做出非平凡的决策，例如为神经算子学习定义合适的输入函数空间。研究发现，最新版本的ChatGPT o3-mini-high通常能提供最准确的结果，并且响应速度比其推理优化版本DeepSeek R1快得多。这种增强的速度和准确性使ChatGPT o3-mini-high目前成为多种计算任务中的更实用和高效的选项。 

---
# Design and implementation of a distributed security threat detection system integrating federated learning and multimodal LLM 

**Title (ZH)**: 集成联邦学习和多模态大语言模型的分布式安全威胁检测系统的设计与实现 

**Authors**: Yuqing Wang, Xiao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17763)  

**Abstract**: Traditional security protection methods struggle to address sophisticated attack vectors in large-scale distributed systems, particularly when balancing detection accuracy with data privacy concerns. This paper presents a novel distributed security threat detection system that integrates federated learning with multimodal large language models (LLMs). Our system leverages federated learning to ensure data privacy while employing multimodal LLMs to process heterogeneous data sources including network traffic, system logs, images, and sensor data. Experimental evaluation on a 10TB distributed dataset demonstrates that our approach achieves 96.4% detection accuracy, outperforming traditional baseline models by 4.1 percentage points. The system reduces both false positive and false negative rates by 1.8 and 2.4 percentage points respectively. Performance analysis shows that our system maintains efficient processing capabilities in distributed environments, requiring 180 seconds for model training and 3.8 seconds for threat detection across the distributed network. These results demonstrate significant improvements in detection accuracy and computational efficiency while preserving data privacy, suggesting strong potential for real-world deployment in large-scale security systems. 

**Abstract (ZH)**: 传统安全防护方法在大规模分布式系统中难以应对复杂的攻击向量，特别是在平衡检测精度与数据隐私之间时。本文提出了一种将联邦学习与多模态大型语言模型（LLMs）集成的新型分布式安全威胁检测系统。该系统利用联邦学习确保数据隐私，并采用多模态LLMs处理包括网络流量、系统日志、图像和传感器数据在内的异构数据源。在包含10TB数据的分布式数据集上的实验评估表明，本方法的检测准确率为96.4%，比传统的基线模型高出4.1个百分点。该系统分别将误报率和漏报率降低了1.8和2.4个百分点。性能分析显示，该系统在分布式环境中保持了高效的处理能力，模型训练需要180秒，分布式网络中的威胁检测需要3.8秒。这些结果展示了在保持数据隐私的同时，在检测精度和计算效率方面取得的重大改进，表明在大规模安全系统中具有很强的实际部署潜力。 

---
# Graded Neural Networks 

**Title (ZH)**: 分层神经网络 

**Authors**: Tony Shaska  

**Link**: [PDF](https://arxiv.org/pdf/2502.17751)  

**Abstract**: This paper presents a novel framework for graded neural networks (GNNs) built over graded vector spaces $\V_\w^n$, extending classical neural architectures by incorporating algebraic grading. Leveraging a coordinate-wise grading structure with scalar action $\lambda \star \x = (\lambda^{q_i} x_i)$, defined by a tuple $\w = (q_0, \ldots, q_{n-1})$, we introduce graded neurons, layers, activation functions, and loss functions that adapt to feature significance. Theoretical properties of graded spaces are established, followed by a comprehensive GNN design, addressing computational challenges like numerical stability and gradient scaling. Potential applications span machine learning and photonic systems, exemplified by high-speed laser-based implementations. This work offers a foundational step toward graded computation, unifying mathematical rigor with practical potential, with avenues for future empirical and hardware exploration. 

**Abstract (ZH)**: 本文提出了一个基于分级向量空间\V_\w^n的新型分级神经网络（GNNs）框架，通过引入代数分级扩展了经典神经架构。利用元组\w = (q_0, \ldots, q_{n-1}) 定义的逐坐标分级结构和标量作用\lambda \star \x = (\lambda^{q_i} x_i)，引入了适应特征重要性的分级神经元、层级、激活函数和损失函数。建立了分级空间的理论性质，并提出了一整套GNN设计，解决了计算挑战如数值稳定性与梯度缩放。潜在应用涉及机器学习和光子系统，以高速激光基实现为例。本文为分级计算奠定了基础步骤，统一了数学严谨性与实际潜力，并为未来的经验和硬件探索提供了途径。 

---
# LLM Inference Acceleration via Efficient Operation Fusion 

**Title (ZH)**: LLM运算融合加速方法 

**Authors**: Mahsa Salmani, Ilya Soloveychik  

**Link**: [PDF](https://arxiv.org/pdf/2502.17728)  

**Abstract**: The rapid development of the Transformer-based Large Language Models (LLMs) in recent years has been closely linked to their ever-growing and already enormous sizes. Many LLMs contain hundreds of billions of parameters and require dedicated hardware resources for training and inference. One of the key challenges inherent to the Transformer architecture is the requirement to support numerous non-linear transformations that involves normalization. For instance, each decoder block typically contains at least one Softmax operation and two Layernorms. The computation of the corresponding normalization scaling factors becomes a major bottleneck as it requires spatial collective operations. In other words, when it comes to the computation of denominators for Softmax and Layernorm, all vector elements must be aggregated into a single location, requiring significant communication. These collective operations slow down inference on Transformers by approximately 20%, defeating the whole purpose of distributed in-memory compute. In this work, we propose an extremely efficient technique that can completely hide the overhead caused by such collective operations. Note that each Softmax and Layernorm operation is typically followed by a linear layer. Since non-linear and linear operations are performed on different hardware engines, they can be easily parallelized once the algebra allows such commutation. By leveraging the inherent properties of linear operations, we can defer the normalization of the preceding Softmax and Layernorm until after the linear layer is computed. Now we can compute the collective scaling factors concurrently with the matrix multiplication and completely hide the latency of the former behind the latter. Such parallelization preserves the numerical accuracy while significantly improving the hardware utilization and reducing the overall latency. 

**Abstract (ZH)**: 基于Transformer的大语言模型的快速发展与其日益庞大的规模密切相关。其中许多模型包含数百亿参数，并需要专用的硬件资源进行训练和推理。Transformer架构的一个核心挑战是需要支持大量的非线性变换，这些变换涉及归一化操作。例如，每个解码器块通常包含至少一个Softmax操作和两个LayerNorm。相应归一化缩放因子的计算成为了一个主要瓶颈，因为它需要空间上的集体操作。换句话说，在Softmax和LayerNorm的分母计算中，所有向量元素必须聚合到一个位置，这需要大量的通信。这些集体操作会将Transformer的推理速度降低约20%，违背了分布式内存计算的初衷。本文提出了一种极其高效的技术，可以完全隐藏由此类集体操作引起的开销。注意到每个Softmax和LayerNorm操作通常后跟一个线性层。由于非线性和线性操作是在不同的硬件引擎上进行的，一旦代数允许交换，它们可以很容易地并行化。通过利用线性操作的固有特性，我们可以在计算线性层之后推迟前面的Softmax和LayerNorm的归一化。现在我们可以在进行矩阵乘法的同时并行计算集体缩放因子，并完全隐藏前者在后者之后的延迟。这种并行化保留了数值精度，显著提高了硬件利用率并减少了总体延迟。 

---
# The GigaMIDI Dataset with Features for Expressive Music Performance Detection 

**Title (ZH)**: GigaMIDI数据集及其用于表达性音乐表演检测的特征 

**Authors**: Keon Ju Maverick Lee, Jeff Ens, Sara Adkins, Pedro Sarmento, Mathieu Barthet, Philippe Pasquier  

**Link**: [PDF](https://arxiv.org/pdf/2502.17726)  

**Abstract**: The Musical Instrument Digital Interface (MIDI), introduced in 1983, revolutionized music production by allowing computers and instruments to communicate efficiently. MIDI files encode musical instructions compactly, facilitating convenient music sharing. They benefit Music Information Retrieval (MIR), aiding in research on music understanding, computational musicology, and generative music. The GigaMIDI dataset contains over 1.4 million unique MIDI files, encompassing 1.8 billion MIDI note events and over 5.3 million MIDI tracks. GigaMIDI is currently the largest collection of symbolic music in MIDI format available for research purposes under fair dealing. Distinguishing between non-expressive and expressive MIDI tracks is challenging, as MIDI files do not inherently make this distinction. To address this issue, we introduce a set of innovative heuristics for detecting expressive music performance. These include the Distinctive Note Velocity Ratio (DNVR) heuristic, which analyzes MIDI note velocity; the Distinctive Note Onset Deviation Ratio (DNODR) heuristic, which examines deviations in note onset times; and the Note Onset Median Metric Level (NOMML) heuristic, which evaluates onset positions relative to metric levels. Our evaluation demonstrates these heuristics effectively differentiate between non-expressive and expressive MIDI tracks. Furthermore, after evaluation, we create the most substantial expressive MIDI dataset, employing our heuristic, NOMML. This curated iteration of GigaMIDI encompasses expressively-performed instrument tracks detected by NOMML, containing all General MIDI instruments, constituting 31% of the GigaMIDI dataset, totalling 1,655,649 tracks. 

**Abstract (ZH)**: MIDI文件的革命：GigaMIDI数据集及其在区分表情与非表情MIDI轨道中的应用 

---
# Solving the Traveling Salesman Problem via Different Quantum Computing Architectures 

**Title (ZH)**: 通过不同量子计算架构解决旅行商问题 

**Authors**: Venkat Padmasola, Zhaotong Li, Rupak Chatterjee, Wesley Dyk  

**Link**: [PDF](https://arxiv.org/pdf/2502.17725)  

**Abstract**: We study the application of emerging photonic and quantum computing architectures to solving the Traveling Salesman Problem (TSP), a well-known NP-hard optimization problem. We investigate several approaches: Simulated Annealing (SA), Quadratic Unconstrained Binary Optimization (QUBO-Ising) methods implemented on quantum annealers and Optical Coherent Ising Machines, as well as the Quantum Approximate Optimization Algorithm (QAOA) and the Quantum Phase Estimation (QPE) algorithm on gate-based quantum computers.
QAOA and QPE were tested on the IBM Quantum platform. The QUBO-Ising method was explored using the D-Wave quantum annealer, which operates on superconducting Josephson junctions, and the QCI Dirac machine, a nonlinear optoelectronic Ising machine. Gate-based quantum computers demonstrated accurate results for small TSP instances in simulation. However, real quantum devices are hindered by noise and limited scalability. Circuit complexity grows with problem size, restricting performance to TSP instances with a maximum of 6 nodes.
In contrast, Ising-based architectures show improved scalability for larger problem sizes. SQUID-based Ising machines can handle TSP instances with up to 12 nodes, while nonlinear optoelectronic Ising machines extend this capability to 18 nodes. Nevertheless, the solutions tend to be suboptimal due to hardware limitations and challenges in achieving ground state convergence as the problem size increases. Despite these limitations, Ising machines demonstrate significant time advantages over classical methods, making them a promising candidate for solving larger-scale TSPs efficiently. 

**Abstract (ZH)**: 我们研究了新兴的光子和量子计算架构在解决旅行商问题（TSP）中的应用，TSP是一个著名的NP难优化问题。我们探讨了几种方法：模拟退火（SA）、在量子退火器和光学相干伊辛机上实现的二次无约束二元优化（QUBO-Ising）方法，以及门基量子计算机上的量子近似优化算法（QAOA）和量子相位估计算法（QPE）。QAOA和QPE在IBM Quantum平台上进行了测试。QUBO-Ising方法在D-Wave量子退火器和QCI Dirac光学非线性光电伊辛机上进行了探索，D-Wave量子退火器基于超导约瑟夫森结工作。门基量子计算机在小规模TSP实例的仿真中展现了准确的结果，但实际量子设备受到噪声和可扩展性限制，随着问题规模的增大，电路复杂度的增加限制了性能，最多只能处理6个节点的问题实例。相比之下，基于伊辛模型的架构对于处理大规模问题表现出更好的可扩展性。基于SQUID的伊辛机能够处理多达12个节点的问题，而光学非线性光电伊辛机将这一能力扩展到18个节点。然而，由于硬件限制和在处理大规模问题时难以达到基态收敛，解决方案往往不是最优的。尽管存在这些限制，伊辛机在解决大规Linux系统问题时显示出显著的时间优势，使其成为解决大规模TSP的有效候选方法。 

---
# Aligning Compound AI Systems via System-level DPO 

**Title (ZH)**: 基于系统级DPO对齐复合AI系统 

**Authors**: Xiangwen Wang, Yibo Jacky Zhang, Zhoujie Ding, Katherine Tsai, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2502.17721)  

**Abstract**: Compound AI systems, comprising multiple interacting components such as LLM agents and external tools, demonstrate state-of-the-art results across diverse tasks. It is hence crucial to align components within the system to produce consistent results that match human expectations. However, conventional alignment methods, such as Direct Preference Optimization (DPO), are not directly applicable to compound AI systems. These challenges include the non-differentiable interactions between components, making end-to-end gradient optimization infeasible. Additionally, system-level preferences cannot be directly translated into component-level preferences, further complicating alignment. We address the issues by formulating compound AI systems as Directed Acyclic Graphs (DAGs), capturing the connections between agents and the data generation processes. We propose a system-level DPO (SysDPO) to jointly align compound systems by adapting the DPO to operate on these DAGs. We study the joint alignment of an LLM and a diffusion model to demonstrate the effectiveness of our approach. Our exploration provides insights into the alignment of compound AI systems and lays a foundation for future advancements. 

**Abstract (ZH)**: 复合AI系统中包含多个相互作用组件（如LLM代理和外部工具）在多种任务中展现出最先进的结果。因此，将系统内的组件调整一致，以产生符合人类预期的结果变得至关重要。然而，传统的对齐方法，如直接偏好优化（DPO），并不适用于复合AI系统。这些挑战包括组件之间的非可微交互，使得端到端的梯度优化不可行。此外，系统级别的偏好无法直接转换为组件级别的偏好，进一步增加了对齐的复杂性。我们通过将复合AI系统形式化为有向无环图（DAGs），捕捉代理之间的连接和数据生成过程，提出了系统级的DPO（SysDPO），以在DAGs上调整DPO来联合对齐复合系统。我们研究了一种LLM与扩散模型的联合对齐，以展示我们方法的有效性。我们的探索为复合AI系统的对齐提供了洞察，并为未来的发展奠定了基础。 

---
# Spontaneous Giving and Calculated Greed in Language Models 

**Title (ZH)**: 自发给予与计算贪婪：语言模型中的表现 

**Authors**: Yuxuan Li, Hirokazu Shirado  

**Link**: [PDF](https://arxiv.org/pdf/2502.17720)  

**Abstract**: Large language models, when trained with reinforcement learning, demonstrate advanced problem-solving capabilities through reasoning techniques like chain of thoughts and reflection. However, it is unclear how these reasoning capabilities extend to social intelligence. In this study, we investigate how reasoning influences model outcomes in social dilemmas. First, we examine the effects of chain-of-thought and reflection techniques in a public goods game. We then extend our analysis to six economic games on cooperation and punishment, comparing off-the-shelf non-reasoning and reasoning models. We find that reasoning models reduce cooperation and norm enforcement, prioritizing individual rationality. Consequently, groups with more reasoning models exhibit less cooperation and lower gains through repeated interactions. These behaviors parallel human tendencies of "spontaneous giving and calculated greed." Our results suggest the need for AI architectures that incorporate social intelligence alongside reasoning capabilities to ensure that AI supports, rather than disrupts, human cooperative intuition. 

**Abstract (ZH)**: 大型语言模型在通过强化学习训练后，通过链式思考和反思等推理技巧展示了高级的问题解决能力。然而，这些推理能力如何扩展到社交智能尚不清楚。本研究探讨了推理对社会困境中模型结果的影响。首先，我们在公共物品游戏中研究链式思考和反思技术的效果。然后，我们将分析扩展到六个涉及合作与惩罚的经济游戏，比较现成的非推理和推理模型。我们发现，推理模型减少了合作和规范执行，优先考虑个体理性。因此，具有更多推理模型的群体在重复互动中表现出更少的合作和更低的收益。这些行为与人类的“自发给予和精心算计的贪婪”倾向相似。我们的研究结果显示，为了确保人工智能支持而非破坏人类的合作直觉，需要并入社交智能的AI架构。 

---
# Bridging Information Gaps with Comprehensive Answers: Improving the Diversity and Informativeness of Follow-Up Questions 

**Title (ZH)**: 填补信息空白 với综合答案：提高后续问题的多样性和信息量 

**Authors**: Zhe Liu, Taekyu Kang, Haoyu Wang, Seyed Hossein Alavi, Vered Shwartz  

**Link**: [PDF](https://arxiv.org/pdf/2502.17715)  

**Abstract**: Effective conversational systems are expected to dynamically generate contextual follow-up questions to elicit new information while maintaining the conversation flow. While humans excel at asking diverse and informative questions by intuitively assessing both obtained and missing information, existing models often fall short of human performance on this task. To mitigate this, we propose a method that generates diverse and informative questions based on targeting unanswered information using a hypothetical LLM-generated "comprehensive answer". Our method is applied to augment an existing follow-up questions dataset. The experimental results demonstrate that language models fine-tuned on the augmented datasets produce follow-up questions of significantly higher quality and diversity. This promising approach could be effectively adopted to future work to augment information-seeking dialogues for reducing ambiguities and improving the accuracy of LLM answers. 

**Abstract (ZH)**: 有效的对话系统应能动态生成与场景相关的问题，以获取新信息并保持对话流畅。虽然人类能够通过直观评估已获得和缺失的信息来提出多样性和信息性的问题，现有模型在这方面往往无法达到人类的表现。为解决这一问题，我们提出了一种方法，该方法基于假设的LLM生成的“全面答案”来针对未解答的信息生成多样性和信息性的问题。该方法应用于扩展现有的后续问题数据集。实验结果表明，针对扩展数据集微调的语言模型生成的后续问题在质量和多样性方面显著提高。这一有前景的方法可以有效地应用于未来工作，以增强信息寻求对话，减少歧义并提高LLM答案的准确性。 

---
# On the usability of generative AI: Human generative AI 

**Title (ZH)**: 关于生成性AI的可用性：人类生成性AI 

**Authors**: Anna Ravera, Cristina Gena  

**Link**: [PDF](https://arxiv.org/pdf/2502.17714)  

**Abstract**: Generative AI systems are transforming content creation, but their usability remains a key challenge. This paper examines usability factors such as user experience, transparency, control, and cognitive load. Common challenges include unpredictability and difficulties in fine-tuning outputs. We review evaluation metrics like efficiency, learnability, and satisfaction, highlighting best practices from various domains. Improving interpretability, intuitive interfaces, and user feedback can enhance usability, making generative AI more accessible and effective. 

**Abstract (ZH)**: 生成式AI系统正在变革内容创作，但其可用性仍是一项关键挑战。本文探讨了包括用户体验、透明度、控制和认知负荷在内的可用性因素。常见挑战包括输出的不可预测性和调优困难。我们回顾了效率、易学性和满意度等评估指标，强调了来自不同领域的最佳实践。提高可解释性、直观的界面和用户反馈可以提升可用性，使生成式AI更加易于使用且效果更佳。 

---
# Contrastive Visual Data Augmentation 

**Title (ZH)**: 对比视觉数据增强 

**Authors**: Yu Zhou, Bingxuan Li, Mohan Tang, Xiaomeng Jin, Te-Lin Wu, Kuan-Hao Huang, Heng Ji, Kai-Wei Chang, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.17709)  

**Abstract**: Large multimodal models (LMMs) often struggle to recognize novel concepts, as they rely on pre-trained knowledge and have limited ability to capture subtle visual details. Domain-specific knowledge gaps in training also make them prone to confusing visually similar, commonly misrepresented, or low-resource concepts. To help LMMs better align nuanced visual features with language, improving their ability to recognize and reason about novel or rare concepts, we propose a Contrastive visual Data Augmentation (CoDA) strategy. CoDA extracts key contrastive textual and visual features of target concepts against the known concepts they are misrecognized as, and then uses multimodal generative models to produce targeted synthetic data. Automatic filtering of extracted features and augmented images is implemented to guarantee their quality, as verified by human annotators. We show the effectiveness and efficiency of CoDA on low-resource concept and diverse scene recognition datasets including INaturalist and SUN. We additionally collect NovelSpecies, a benchmark dataset consisting of newly discovered animal species that are guaranteed to be unseen by LMMs. LLaVA-1.6 1-shot updating results on these three datasets show CoDA significantly improves SOTA visual data augmentation strategies by 12.3% (NovelSpecies), 5.1% (SUN), and 6.0% (iNat) absolute gains in accuracy. 

**Abstract (ZH)**: Large 多模态模型（LMMs）往往难以识别新的概念，因为它们依赖于预训练的知识，并且在捕捉细微的视觉细节方面能力有限。训练中的领域特定知识差距也使它们容易混淆视觉上相似、常见误表征或低资源的概念。为了帮助 LMMs 更好地将复杂的视觉特征与语言对齐，提高它们识别和推理新或稀有概念的能力，我们提出了一种对比视觉数据增强（CoDA）策略。CoDA 从目标概念与它们被误识别为的已知概念的关键对比文本和视觉特征中提取，然后使用多模态生成模型生成针对性的合成数据。通过人工注释者验证，实现了自动提取特征和增强图像的质量过滤。我们在包含 INaturalist 和 SUN 的低资源概念和多样场景识别数据集上展示了 CoDA 的有效性与效率。此外，我们收集了 NovelSpecies，这是一个基准数据集，其中包含确保 LMMs 未见过的新发现动物物种。对于这三个数据集上的 LLava-1.6 一次-shot 更新结果，CoDA 在准确性上分别提高了 NovelSpecies 12.3%、SUN 5.1% 和 iNat 6.0%。 

---
# To Patch or Not to Patch: Motivations, Challenges, and Implications for Cybersecurity 

**Title (ZH)**: 要不要打补丁：动机、挑战及网络安全影响 

**Authors**: Jason R. C. Nurse  

**Link**: [PDF](https://arxiv.org/pdf/2502.17703)  

**Abstract**: As technology has become more embedded into our society, the security of modern-day systems is paramount. One topic which is constantly under discussion is that of patching, or more specifically, the installation of updates that remediate security vulnerabilities in software or hardware systems. This continued deliberation is motivated by complexities involved with patching; in particular, the various incentives and disincentives for organizations and their cybersecurity teams when deciding whether to patch. In this paper, we take a fresh look at the question of patching and critically explore why organizations and IT/security teams choose to patch or decide against it (either explicitly or due to inaction). We tackle this question by aggregating and synthesizing prominent research and industry literature on the incentives and disincentives for patching, specifically considering the human aspects in the context of these motives. Through this research, this study identifies key motivators such as organizational needs, the IT/security team's relationship with vendors, and legal and regulatory requirements placed on the business and its staff. There are also numerous significant reasons discovered for why the decision is taken not to patch, including limited resources (e.g., person-power), challenges with manual patch management tasks, human error, bad patches, unreliable patch management tools, and the perception that related vulnerabilities would not be exploited. These disincentives, in combination with the motivators above, highlight the difficult balance that organizations and their security teams need to maintain on a daily basis. Finally, we conclude by discussing implications of these findings and important future considerations. 

**Abstract (ZH)**: 随着技术越来越深入融入我们的社会，现代系统的安全性变得至关重要。关于补丁安装的问题，尤其是软件或硬件系统中安全漏洞修复的更新安装，一直是持续讨论的焦点。这种持续的讨论受到补丁安装复杂性的推动；特别是组织及其网络安全团队在决定是否安装补丁时的各种动机和抑制因素。本文从一个新的角度探讨补丁安装问题，并深入探讨组织和IT/安全团队选择安装补丁或不安装补丁（无论是明确选择不安装还是因无行动而未安装）的原因。我们通过综合和合成有关补丁安装动机和抑制因素的主要研究和行业文献，并特别考虑这些动机的人文方面，来回答这个问题。通过这项研究，本文确定了关键的动机，如组织需求、IT/安全团队与供应商的关系以及对企业和员工的法律和监管要求。还发现了许多不安装补丁的重要原因，包括资源有限（例如人力）、手动补丁管理任务的挑战、人为错误、不良补丁、不可靠的补丁管理工具以及认为相关漏洞不会被利用的感知。这些抑制因素与上述动机的结合，突显了组织及其安全团队在日常工作中需要维持的困难平衡。最后，本文讨论了这些发现的影响和重要的未来考虑。 

---
# Yes, Q-learning Helps Offline In-Context RL 

**Title (ZH)**: 是的，Q-learning有助于离线场景内的RL学习。 

**Authors**: Denis Tarasov, Alexander Nikulin, Ilya Zisman, Albina Klepach, Andrei Polubarov, Nikita Lyubaykin, Alexander Derevyagin, Igor Kiselev, Vladislav Kurenkov  

**Link**: [PDF](https://arxiv.org/pdf/2502.17666)  

**Abstract**: In this work, we explore the integration of Reinforcement Learning (RL) approaches within a scalable offline In-Context RL (ICRL) framework. Through experiments across more than 150 datasets derived from GridWorld and MuJoCo environments, we demonstrate that optimizing RL objectives improves performance by approximately 40% on average compared to the widely established Algorithm Distillation (AD) baseline across various dataset coverages, structures, expertise levels, and environmental complexities. Our results also reveal that offline RL-based methods outperform online approaches, which are not specifically designed for offline scenarios. These findings underscore the importance of aligning the learning objectives with RL's reward-maximization goal and demonstrate that offline RL is a promising direction for application in ICRL settings. 

**Abstract (ZH)**: 本研究探讨了在可扩展的离线情境强化学习(ICRL)框架中集成强化学习(RL)方法的可能性。通过跨越150多个源自GridWorld和MuJoCo环境的数据集的实验，我们证明与广泛采用的算法蒸馏(AD)基线相比，优化RL目标可以平均提高约40%的性能，这一改善在不同数据集覆盖度、结构、专家水平和环境复杂性下均有表现。研究结果还表明，基于离线RL的方法在一般不为离线场景设计的在线方法之上表现出色。这些发现强调了将学习目标与RL的奖励最大化目标对齐的重要性，并展示了在ICRL设置中应用离线RL是一个有前景的方向。 

---
# Effective Field Neural Network 

**Title (ZH)**: 有效场神经网络 

**Authors**: Xi Liu, Yujun Zhao, Chun Yu Wan, Yang Zhang, Junwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17665)  

**Abstract**: In recent years, with the rapid development of machine learning, physicists have been exploring its new applications in solving or alleviating the curse of dimensionality in many-body problems. In order to accurately reflect the underlying physics of the problem, domain knowledge must be encoded into the machine learning algorithms. In this work, inspired by field theory, we propose a new set of machine learning models called effective field neural networks (EFNNs) that can automatically and efficiently capture important many-body interactions through multiple self-refining processes. Taking the classical $3$-spin infinite-range model and the quantum double exchange model as case studies, we explicitly demonstrate that EFNNs significantly outperform fully-connected deep neural networks (DNNs) and the effective model. Furthermore, with the help of convolution operations, the EFNNs learned in a small system can be seamlessly used in a larger system without additional training and the relative errors even decrease, which further demonstrates the efficacy of EFNNs in representing core physical behaviors. 

**Abstract (ZH)**: 近年来，随着机器学习的快速發展，物理学家在探索将其应用于解決或緩解多體問題的維度災難方面取得了新進展。為了準確反映問題的基礎物理特性，必須將領域知識編碼到機器學習算法中。在本工作中，受到場論的思想啟發，我們提出了一種新的機器學習模型有效場神經網絡（EFNNs），該模型能夠通過多個自反修諫過程自動且高效地捕獲重要多體交互作用。以经典的3自旋无穷范围模型和量子双交换模型为例，我们Explicitly展示了EFNNs显著优于全连接深度神经网络（DNNs）和有效模型。此外，借助卷积操作，EFNNs可以在小系统中学习后无缝应用于大系统中，而无需额外训练，相对误差甚至减小，这进一步证明了EFNNs在表示核心物理行为方面的有效性。 

---
# StatLLM: A Dataset for Evaluating the Performance of Large Language Models in Statistical Analysis 

**Title (ZH)**: StatLLM：用于评估大型语言模型在统计分析性能的数据集 

**Authors**: Xinyi Song, Lina Lee, Kexin Xie, Xueying Liu, Xinwei Deng, Yili Hong  

**Link**: [PDF](https://arxiv.org/pdf/2502.17657)  

**Abstract**: The coding capabilities of large language models (LLMs) have opened up new opportunities for automatic statistical analysis in machine learning and data science. However, before their widespread adoption, it is crucial to assess the accuracy of code generated by LLMs. A major challenge in this evaluation lies in the absence of a benchmark dataset for statistical code (e.g., SAS and R). To fill in this gap, this paper introduces StatLLM, an open-source dataset for evaluating the performance of LLMs in statistical analysis. The StatLLM dataset comprises three key components: statistical analysis tasks, LLM-generated SAS code, and human evaluation scores. The first component includes statistical analysis tasks spanning a variety of analyses and datasets, providing problem descriptions, dataset details, and human-verified SAS code. The second component features SAS code generated by ChatGPT 3.5, ChatGPT 4.0, and Llama 3.1 for those tasks. The third component contains evaluation scores from human experts in assessing the correctness, effectiveness, readability, executability, and output accuracy of the LLM-generated code. We also illustrate the unique potential of the established benchmark dataset for (1) evaluating and enhancing natural language processing metrics, (2) assessing and improving LLM performance in statistical coding, and (3) developing and testing of next-generation statistical software - advancements that are crucial for data science and machine learning research. 

**Abstract (ZH)**: 大型语言模型（LLMs）的编码能力为机器学习和数据科学中的自动统计分析打开了新的机会。然而，在广泛采用之前，评估LLMs生成代码的准确性至关重要。鉴于此，本文引入了StatLLM，这是一个开源数据集，用于评估LLMs在统计分析中的性能。StatLLM数据集包括三个关键组件：统计分析任务、LLM生成的SAS代码以及人类评估得分。该数据集的第一个组件包括涵盖多种分析和数据集的统计分析任务，提供问题描述、数据集细节和人工验证的SAS代码。第二个组件展示了ChatGPT 3.5、ChatGPT 4.0和Llama 3.1为这些任务生成的SAS代码。第三个组件包含专家对LLM生成代码的正确性、有效性、可读性、可执行性和输出准确性的人类评估得分。此外，本文还阐述了已建立基准数据集的独特潜力，用于（1）评估和提升自然语言处理指标，（2）评估和提升LLMs在统计编码中的性能，以及（3）开发和测试下一代统计软件——这些都是数据科学和机器学习研究的关键进展。 

---
# METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling 

**Title (ZH)**: METAL：一种用于图表生成的多agent框架，具备测试时扩展能力 

**Authors**: Bingxuan Li, Yiwei Wang, Jiuxiang Gu, Kai-Wei Chang, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.17651)  

**Abstract**: Chart generation aims to generate code to produce charts satisfying the desired visual properties, e.g., texts, layout, color, and type. It has great potential to empower the automatic professional report generation in financial analysis, research presentation, education, and healthcare. In this work, we build a vision-language model (VLM) based multi-agent framework for effective automatic chart generation. Generating high-quality charts requires both strong visual design skills and precise coding capabilities that embed the desired visual properties into code. Such a complex multi-modal reasoning process is difficult for direct prompting of VLMs. To resolve these challenges, we propose METAL, a multi-agent framework that decomposes the task of chart generation into the iterative collaboration among specialized agents. METAL achieves 5.2% improvement in accuracy over the current best result in the chart generation task. The METAL framework exhibits the phenomenon of test-time scaling: its performance increases monotonically as the logarithmic computational budget grows from 512 to 8192 tokens. In addition, we find that separating different modalities during the critique process of METAL boosts the self-correction capability of VLMs in the multimodal context. 

**Abstract (ZH)**: 基于视觉-语言模型的多agent框架下的图表生成方法 

---
# Wearable Meets LLM for Stress Management: A Duoethnographic Study Integrating Wearable-Triggered Stressors and LLM Chatbots for Personalized Interventions 

**Title (ZH)**: 可穿戴设备结合大语言模型进行压力管理：一种整合可穿戴设备触发的压力源和大语言模型聊天机器人的个性化干预 dúoyíngnuòlùn jiājié héqiǎojiǎngxínɡ de yālì guǎnlǐ：yī zhǒnɡ hélián kěwaklıān bèiqiè tífù de yālì yuán hé dàyǎ jiǎnɡxínɡ mó’érlénɡ de ɡè rèn huīnéng ɡānjūn dúoyíng 

**Authors**: Sameer Neupane, Poorvesh Dongre, Denis Gracanin, Santosh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.17650)  

**Abstract**: We use a duoethnographic approach to study how wearable-integrated LLM chatbots can assist with personalized stress management, addressing the growing need for immediacy and tailored interventions. Two researchers interacted with custom chatbots over 22 days, responding to wearable-detected physiological prompts, recording stressor phrases, and using them to seek tailored interventions from their LLM-powered chatbots. They recorded their experiences in autoethnographic diaries and analyzed them during weekly discussions, focusing on the relevance, clarity, and impact of chatbot-generated interventions. Results showed that even though most events triggered by the wearable were meaningful, only one in five warranted an intervention. It also showed that interventions tailored with brief event descriptions were more effective than generic ones. By examining the intersection of wearables and LLM, this research contributes to developing more effective, user-centric mental health tools for real-time stress relief and behavior change. 

**Abstract (ZH)**: 我们采用双重民族志方法研究穿戴设备集成的大语言模型聊天机器人在个人压力管理中的辅助作用，以应对即时性和个性化干预日益增长的需求。两名研究人员在22天内与定制聊天机器人互动，响应穿戴设备检测到的生理信号提示、记录压力源短语，并使用这些信息寻求基于大语言模型的聊天机器人提供的个性化干预措施。他们记录了自己的体验并在每周的讨论中进行分析，重点关注聊天机器人生成的干预措施的相关性、清晰度和影响。研究结果显示，尽管大多数由穿戴设备触发的事件具有重要意义，但仅五分之一的事件需要干预。同时发现，使用简要事件描述定制的干预措施比通用的干预措施更有效。通过探讨穿戴设备与大语言模型的交集，本研究为开发更有效、以用户为中心的实时减压和行为改变的心理健康工具做出了贡献。 

---
# Requirements for Quality Assurance of AI Models for Early Detection of Lung Cancer 

**Title (ZH)**: AI模型早期检测肺癌的质量保证要求 

**Authors**: Horst K. Hahn, Matthias S. May, Volker Dicken, Michael Walz, Rainer Eßeling, Bianca Lassen-Schmidt, Robert Rischen, Jens Vogel-Claussen, Konstantin Nikolaou, Jörg Barkhausen  

**Link**: [PDF](https://arxiv.org/pdf/2502.17639)  

**Abstract**: Lung cancer is the second most common cancer and the leading cause of cancer-related deaths worldwide. Survival largely depends on tumor stage at diagnosis, and early detection with low-dose CT can significantly reduce mortality in high-risk patients. AI can improve the detection, measurement, and characterization of pulmonary nodules while reducing assessment time. However, the training data, functionality, and performance of available AI systems vary considerably, complicating software selection and regulatory evaluation. Manufacturers must specify intended use and provide test statistics, but they can choose their training and test data, limiting standardization and comparability. Under the EU AI Act, consistent quality assurance is required for AI-based nodule detection, measurement, and characterization.
This position paper proposes systematic quality assurance grounded in a validated reference dataset, including real screening cases plus phantom data to verify volume and growth rate measurements. Regular updates shall reflect demographic shifts and technological advances, ensuring ongoing relevance. Consequently, ongoing AI quality assurance is vital. Regulatory challenges are also adressed. While the MDR and the EU AI Act set baseline requirements, they do not adequately address self-learning algorithms or their updates. A standardized, transparent quality assessment - based on sensitivity, specificity, and volumetric accuracy - enables an objective evaluation of each AI solution's strengths and weaknesses. Establishing clear testing criteria and systematically using updated reference data lay the groundwork for comparable performance metrics, informing tenders, guidelines, and recommendations. 

**Abstract (ZH)**: 肺癌是全球第二常见的癌症，并且是癌症相关死亡的主要原因。生存率主要取决于诊断时的肿瘤分期，早期通过低剂量CT检测可显著降低高风险患者的死亡率。AI可以提高肺结节的检测、测量和表征能力，同时减少评估时间。然而，现有AI系统的训练数据、功能和性能差异较大，这给软件选择和监管评估带来了复杂性。制造商必须明确预期用途并提供测试统计，但可以自行选择训练和测试数据，这限制了标准化和可比性。根据《欧盟AI法案》，基于验证参考数据（包括真实的筛查病例及模拟数据以验证体积和生长率测量）的一致质量保证是必需的。

本文提出了基于验证参考数据的系统质量保证，包括真实筛查病例和模拟数据，以验证体积和生长率测量。定期更新应反映人口结构变化和技术进步，以确保持续相关性。因此，持续的AI质量保证至关重要。本文还解决了监管挑战。虽然MDR和《欧盟AI法案》设定了基本要求，但并未充分解决自学习算法及其更新的问题。基于灵敏度、特异性和容积准确性进行的标准化、透明的质量评估能够客观评价每个AI解决方案的优势和不足。明确测试标准并系统使用更新的参考数据，为可比性能指标奠定了基础，有助于标书、指南和建议的制定。 

---
# Towards Robust Legal Reasoning: Harnessing Logical LLMs in Law 

**Title (ZH)**: 面向稳健的法律推理：利用逻辑大语言模型来推动法律领域的发展 

**Authors**: Manuj Kant, Sareh Nabi, Manav Kant, Roland Scharrer, Megan Ma, Marzieh Nabi  

**Link**: [PDF](https://arxiv.org/pdf/2502.17638)  

**Abstract**: Legal services rely heavily on text processing. While large language models (LLMs) show promise, their application in legal contexts demands higher accuracy, repeatability, and transparency. Logic programs, by encoding legal concepts as structured rules and facts, offer reliable automation, but require sophisticated text extraction. We propose a neuro-symbolic approach that integrates LLMs' natural language understanding with logic-based reasoning to address these limitations.
As a legal document case study, we applied neuro-symbolic AI to coverage-related queries in insurance contracts using both closed and open-source LLMs. While LLMs have improved in legal reasoning, they still lack the accuracy and consistency required for complex contract analysis. In our analysis, we tested three methodologies to evaluate whether a specific claim is covered under a contract: a vanilla LLM, an unguided approach that leverages LLMs to encode both the contract and the claim, and a guided approach that uses a framework for the LLM to encode the contract. We demonstrated the promising capabilities of LLM + Logic in the guided approach. 

**Abstract (ZH)**: 法律服务高度依赖文本处理。虽然大型语言模型（LLMs）显示出潜力，但在法律情境中的应用需更高的准确度、可重复性和透明度。通过将法律概念编码为结构化规则和事实的逻辑程序提供了可靠的自动化，但需要复杂的文本提取。我们提出了一种神经符号方法，将LLMs的自然语言理解与基于逻辑的推理相结合，以解决这些限制。

作为法律文件案例研究，我们使用闭源和开源LLMs将神经符号AI应用于保险合同的覆盖查询中。虽然LLMs在法律推理方面有所改进，但仍缺乏复杂合同分析所需的准确性和一致性。在我们的分析中，我们测试了三种方法来评估特定索赔是否在合同范围内： Vanilla LLM，一种未引导的方法，利用LLMs编码合同和索赔，以及一种引导方法，使用框架让LLMs编码合同。我们展示了引导方法中LLM + Logic的有前景的能力。 

---
# Theory-guided Pseudo-spectral Full Waveform Inversion via Deep Neural Networks 

**Title (ZH)**: 基于理论指导的伪谱全波形反演深度神经网络方法 

**Authors**: Christopher Zerafa, Pauline Galea, Cristiana Sebu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17624)  

**Abstract**: Full-Waveform Inversion seeks to achieve a high-resolution model of the subsurface through the application of multi-variate optimization to the seismic inverse problem. Although now a mature technology, FWI has limitations related to the choice of the appropriate solver for the forward problem in challenging environments requiring complex assumptions, and very wide angle and multi-azimuth data necessary for full reconstruction are often not available.
Deep Learning techniques have emerged as excellent optimization frameworks. Data-driven methods do not impose a wave propagation model and are not exposed to modelling errors. On the contrary, deterministic models are governed by the laws of physics.
Seismic FWI has recently started to be investigated as a Deep Learning framework. Focus has been on the time-domain, while the pseudo-spectral domain has not been yet explored. However, classical FWI experienced major breakthroughs when pseudo-spectral approaches were employed. This work addresses the lacuna that exists in incorporating the pseudo-spectral approach within Deep Learning. This has been done by re-formulating the pseudo-spectral FWI problem as a Deep Learning algorithm for a theory-driven pseudo-spectral approach. A novel Recurrent Neural Network framework is proposed. This is qualitatively assessed on synthetic data, applied to a two-dimensional Marmousi dataset and evaluated against deterministic and time-based approaches.
Pseudo-spectral theory-guided FWI using RNN was shown to be more accurate than classical FWI with only 0.05 error tolerance and 1.45\% relative percent-age error. Indeed, this provides more stable convergence, able to identify faults better and has more low frequency content than classical FWI. Moreover, RNN was more suited than classical FWI at edge detection in the shallow and deep sections due to cleaner receiver residuals. 

**Abstract (ZH)**: 全波形 inversion基于多变量优化解决地震逆问题以实现地下高分辨率模型。尽管全波形 inversion已成为一种成熟的技術，但在复杂环境下，适当前向问题求解器的选择仍是其局限性之一。全貌和多方位数据对于完整重建往往是不可得的。

深度学习技术已经发展成为优秀的优化框架。数据驱动的方法无需设定波传播模型，也不会受到模型误差的影响。相反，确定性模型则遵循物理定律。

地震全波形 inversion最近开始作为深度学习框架进行研究。研究主要集中在时域，而伪谱域尚未被探索。然而，经典全波形 inversion在伪谱方法被采用时经历了重大突破。本工作填补了将伪谱方法融入深度学习的空白。通过将伪谱全波形 inversion问题重新表述为理论导向的伪谱深度学习算法，提出了一种新颖的循环神经网络框架。该方法在合成数据上进行了定性评估，并应用于二维Marmousi数据集，与确定性和基于时间的方法进行了对比。

基于RNN的伪谱理论导向全波形 inversion表现优于仅采用0.05误差容限和1.45%相对误差的经典全波形 inversion，提供了更稳定的收敛性，能够更好地识别断层，并具有更多的低频成分。此外，RNN在浅部和深部区块边缘检测方面比经典全波形 inversion更具优势，因为其接收器残差更干净。 

---
# Hierarchical Imitation Learning of Team Behavior from Heterogeneous Demonstrations 

**Title (ZH)**: 异质示范指导下分层模仿学习团队行为 

**Authors**: Sangwon Seo, Vaibhav Unhelkar  

**Link**: [PDF](https://arxiv.org/pdf/2502.17618)  

**Abstract**: Successful collaboration requires team members to stay aligned, especially in complex sequential tasks. Team members must dynamically coordinate which subtasks to perform and in what order. However, real-world constraints like partial observability and limited communication bandwidth often lead to suboptimal collaboration. Even among expert teams, the same task can be executed in multiple ways. To develop multi-agent systems and human-AI teams for such tasks, we are interested in data-driven learning of multimodal team behaviors. Multi-Agent Imitation Learning (MAIL) provides a promising framework for data-driven learning of team behavior from demonstrations, but existing methods struggle with heterogeneous demonstrations, as they assume that all demonstrations originate from a single team policy. Hence, in this work, we introduce DTIL: a hierarchical MAIL algorithm designed to learn multimodal team behaviors in complex sequential tasks. DTIL represents each team member with a hierarchical policy and learns these policies from heterogeneous team demonstrations in a factored manner. By employing a distribution-matching approach, DTIL mitigates compounding errors and scales effectively to long horizons and continuous state representations. Experimental results show that DTIL outperforms MAIL baselines and accurately models team behavior across a variety of collaborative scenarios. 

**Abstract (ZH)**: 成功的协作需要团队成员保持一致，特别是在复杂的序列任务中。团队成员必须动态协调执行哪些亚任务及其顺序。然而，现实世界中的部分可观测性和有限的通信带宽常导致协作效果不佳。即使是专家团队，相同的任务也可以采用多种方式执行。为了开发此类任务的多智能体系统和人机团队，我们对数据驱动的多模态团队行为学习感兴趣。多智能体模仿学习（MAIL）为从示范中学习团队行为的数据驱动学习提供了有前途的框架，但现有方法在处理异构示范时遇到了困难，因为它们假定所有示范都源自一个单一的团队策略。因此，在这项工作中，我们引入了DTIL：一种用于复杂序列任务中学习多模态团队行为的层次MAIL算法。DTIL通过层次政策表示每个团队成员，并以分解的方式从异构团队示范中学习这些政策。通过采用分布匹配方法，DTIL减轻了累积误差，并有效地扩展到长期限和连续状态表示中。实验结果显示，DTIL优于MAIL基线，并且能够准确地建模多种协作场景下的团队行为。 

---
# Flexible Counterfactual Explanations with Generative Models 

**Title (ZH)**: 基于生成模型的灵活反事实解释 

**Authors**: Stig Hellemans, Andres Algaba, Sam Verboven, Vincent Ginis  

**Link**: [PDF](https://arxiv.org/pdf/2502.17613)  

**Abstract**: Counterfactual explanations provide actionable insights to achieve desired outcomes by suggesting minimal changes to input features. However, existing methods rely on fixed sets of mutable features, which makes counterfactual explanations inflexible for users with heterogeneous real-world constraints. Here, we introduce Flexible Counterfactual Explanations, a framework incorporating counterfactual templates, which allows users to dynamically specify mutable features at inference time. In our implementation, we use Generative Adversarial Networks (FCEGAN), which align explanations with user-defined constraints without requiring model retraining or additional optimization. Furthermore, FCEGAN is designed for black-box scenarios, leveraging historical prediction datasets to generate explanations without direct access to model internals. Experiments across economic and healthcare datasets demonstrate that FCEGAN significantly improves counterfactual explanations' validity compared to traditional benchmark methods. By integrating user-driven flexibility and black-box compatibility, counterfactual templates support personalized explanations tailored to user constraints. 

**Abstract (ZH)**: 灵活的反事实解释：结合反事实模板以适应异质实际约束的框架 

---
# SynthRAD2025 Grand Challenge dataset: generating synthetic CTs for radiotherapy 

**Title (ZH)**: SynthRAD2025 大挑战数据集：用于放射治疗的合成CT图像生成 

**Authors**: Adrian Thummerer, Erik van der Bijl, Arthur Jr Galapon, Florian Kamp, Mark Savenije, Christina Muijs, Shafak Aluwini, Roel J.H.M. Steenbakkers, Stephanie Beuel, Martijn P.W. Intven, Johannes A. Langendijk, Stefan Both, Stefanie Corradini, Viktor Rogowski, Maarten Terpstra, Niklas Wahl, Christopher Kurz, Guillaume Landry, Matteo Maspero  

**Link**: [PDF](https://arxiv.org/pdf/2502.17609)  

**Abstract**: Medical imaging is essential in modern radiotherapy, supporting diagnosis, treatment planning, and monitoring. Synthetic imaging, particularly synthetic computed tomography (sCT), is gaining traction in radiotherapy. The SynthRAD2025 dataset and Grand Challenge promote advancements in sCT generation by providing a benchmarking platform for algorithms using cone-beam CT (CBCT) and magnetic resonance imaging (MRI).
The dataset includes 2362 cases: 890 MRI-CT and 1472 CBCT-CT pairs from head-and-neck, thoracic, and abdominal cancer patients treated at five European university medical centers (UMC Groningen, UMC Utrecht, Radboud UMC, LMU University Hospital Munich, and University Hospital of Cologne). Data were acquired with diverse scanners and protocols. Pre-processing, including rigid and deformable image registration, ensures high-quality, modality-aligned images. Extensive quality assurance validates image consistency and usability.
All imaging data is provided in MetaImage (.mha) format, ensuring compatibility with medical image processing tools. Metadata, including acquisition parameters and registration details, is available in structured CSV files. To maintain dataset integrity, SynthRAD2025 is divided into training (65%), validation (10%), and test (25%) sets. The dataset is accessible at this https URL under the SynthRAD2025 collection.
This dataset supports benchmarking and the development of synthetic imaging techniques for radiotherapy applications. Use cases include sCT generation for MRI-only and MR-guided photon/proton therapy, CBCT-based dose calculations, and adaptive radiotherapy workflows. By integrating diverse acquisition settings, SynthRAD2025 fosters robust, generalizable image synthesis algorithms, advancing personalized cancer care and adaptive radiotherapy. 

**Abstract (ZH)**: 医学成像是现代放射治疗中的重要组成部分，支持诊断、治疗计划和监测。合成影像，特别是合成CT（sCT），在放射治疗中逐渐受到重视。SynthRAD2025数据集和大赛通过提供基于锥束CT（CBCT）和磁共振成像（MRI）算法的基准平台，促进了sCT生成技术的发展。 

---
# Data-Driven Pseudo-spectral Full Waveform Inversion via Deep Neural Networks 

**Title (ZH)**: 基于深度神经网络的数据驱动伪谱全波形反演 

**Authors**: Christopher Zerafa, Pauline Galea, Cristiana Sebu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17608)  

**Abstract**: FWI seeks to achieve a high-resolution model of the subsurface through the application of multi-variate optimization to the seismic inverse problem. Although now a mature technology, FWI has limitations related to the choice of the appropriate solver for the forward problem in challenging environments requiring complex assumptions, and very wide angle and multi-azimuth data necessary for full reconstruction are often not available.
Deep Learning techniques have emerged as excellent optimization frameworks. These exist between data and theory-guided methods. Data-driven methods do not impose a wave propagation model and are not exposed to modelling errors. On the contrary, deterministic models are governed by the laws of physics.
Application of seismic FWI has recently started to be investigated within Deep Learning. This has focussed on the time-domain approach, while the pseudo-spectral domain has not been yet explored. However, classical FWI experienced major breakthroughs when pseudo-spectral approaches were employed. This work addresses the lacuna that exists in incorporating the pseudo-spectral approach within Deep Learning. This has been done by re-formulating the pseudo-spectral FWI problem as a Deep Learning algorithm for a data-driven pseudo-spectral approach. A novel DNN framework is proposed. This is formulated theoretically, qualitatively assessed on synthetic data, applied to a two-dimensional Marmousi dataset and evaluated against deterministic and time-based approaches.
Inversion of data-driven pseudo-spectral DNN was found to outperform classical FWI for deeper and over-thrust areas. This is due to the global approximator nature of the technique and hence not bound by forward-modelling physical constraints from ray-tracing. 

**Abstract (ZH)**: 基于伪谱的深度学习全波场逆问题方法 

---
# PICASO: Permutation-Invariant Context Composition with State Space Models 

**Title (ZH)**: PICASO: 基于状态空间模型的排列不变上下文组合 

**Authors**: Tian Yu Liu, Alessandro Achille, Matthew Trager, Aditya Golatkar, Luca Zancato, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2502.17605)  

**Abstract**: Providing Large Language Models with relevant contextual knowledge at inference time has been shown to greatly improve the quality of their generations. This is often achieved by prepending informative passages of text, or 'contexts', retrieved from external knowledge bases to their input. However, processing additional contexts online incurs significant computation costs that scale with their length. State Space Models (SSMs) offer a promising solution by allowing a database of contexts to be mapped onto fixed-dimensional states from which to start the generation. A key challenge arises when attempting to leverage information present across multiple contexts, since there is no straightforward way to condition generation on multiple independent states in existing SSMs. To address this, we leverage a simple mathematical relation derived from SSM dynamics to compose multiple states into one that efficiently approximates the effect of concatenating textual contexts. Since the temporal ordering of contexts can often be uninformative, we enforce permutation-invariance by efficiently averaging states obtained via our composition algorithm across all possible context orderings. We evaluate our resulting method on WikiText and MSMARCO in both zero-shot and fine-tuned settings, and show that we can match the strongest performing baseline while enjoying on average 5.4x speedup. 

**Abstract (ZH)**: 为大型语言模型在推理时提供相关的上下文知识已被证明能显著提高其生成质量。这通常通过在输入前添加来自外部知识库的信息性文本片段或“上下文”来实现。然而，实时处理额外的上下文会带来显著的计算成本，这些成本随着上下文长度的增加而增加。状态空间模型（SSMs）通过允许将上下文数据库映射到固定维度的状态，从而从中开始生成，提供了颇具前景的解决方案。当尝试利用多个上下文中存在但无关紧要的信息时，一个关键挑战在于无法直接根据不同独立状态对生成进行条件约束。为解决这一问题，我们利用SSM动力学中的一种简单数学关系，将多个状态组合成一个能够高效模拟文本上下文串联效果的状态。由于上下文的时间顺序往往无关紧要，我们通过对通过组合算法获得的状态进行高效平均，来强制状态不变性，从而考虑所有可能的上下文顺序。我们在WikiText和MSMARCO上分别在零样本和微调设置中评估了该方法，结果显示，我们能够匹配表现最强的基线模型，同时平均加速5.4倍。 

---
# Hallucination Detection in LLMs Using Spectral Features of Attention Maps 

**Title (ZH)**: LLMs中基于注意力图谱特征的幻觉检测 

**Authors**: Jakub Binkowski, Denis Janiak, Albert Sawczyn, Bogdan Gabrys, Tomasz Kajdanowicz  

**Link**: [PDF](https://arxiv.org/pdf/2502.17598)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various tasks but remain prone to hallucinations. Detecting hallucinations is essential for safety-critical applications, and recent methods leverage attention map properties to this end, though their effectiveness remains limited. In this work, we investigate the spectral features of attention maps by interpreting them as adjacency matrices of graph structures. We propose the $\text{LapEigvals}$ method, which utilises the top-$k$ eigenvalues of the Laplacian matrix derived from the attention maps as an input to hallucination detection probes. Empirical evaluations demonstrate that our approach achieves state-of-the-art hallucination detection performance among attention-based methods. Extensive ablation studies further highlight the robustness and generalisation of $\text{LapEigvals}$, paving the way for future advancements in the hallucination detection domain. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务中表现出色，但仍易产生幻觉。检测幻觉对于安全性关键应用至关重要，尽管近期方法通过利用注意力图的特性来实现这一目标，但其有效性仍有限。在本工作中，我们通过将注意力图解释为图结构的相邻矩阵来研究其频谱特征。我们提出了LapEigvals方法，该方法利用从注意力图导出的拉普拉斯矩阵的前k个特征值作为幻觉检测探针的输入。实验评估表明，我们的方法在基于注意力的方法中实现了最先进的幻觉检测性能。广泛的消融研究进一步突显了LapEigvals的鲁棒性和泛化能力，为幻觉检测领域的未来进步铺平了道路。 

---
# Synergizing Deep Learning and Full-Waveform Inversion: Bridging Data-Driven and Theory-Guided Approaches for Enhanced Seismic Imaging 

**Title (ZH)**: 深度融合学习与全波形反演：数据驱动与理论指导相结合的地震成像增强方法 

**Authors**: Christopher Zerafa, Pauline Galea, Cristiana Sebu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17585)  

**Abstract**: This review explores the integration of deep learning (DL) with full-waveform inversion (FWI) for enhanced seismic imaging and subsurface characterization. It covers FWI and DL fundamentals, geophysical applications (velocity estimation, deconvolution, tomography), and challenges (model complexity, data quality). The review also outlines future research directions, including hybrid, generative, and physics-informed models for improved accuracy, efficiency, and reliability in subsurface property estimation. The synergy between DL and FWI has the potential to transform geophysics, providing new insights into Earth's subsurface. 

**Abstract (ZH)**: 深学习与全波形反演集成在地震成像与地下表征中的研究综述 

---
# Training a Generally Curious Agent 

**Title (ZH)**: 训练一个普遍好奇的智能体 

**Authors**: Fahim Tajwar, Yiding Jiang, Abitha Thankaraj, Sumaita Sadia Rahman, J Zico Kolter, Jeff Schneider, Ruslan Salakhutdinov  

**Link**: [PDF](https://arxiv.org/pdf/2502.17543)  

**Abstract**: Efficient exploration is essential for intelligent systems interacting with their environment, but existing language models often fall short in scenarios that require strategic information gathering. In this paper, we present PAPRIKA, a fine-tuning approach that enables language models to develop general decision-making capabilities that are not confined to particular environments. By training on synthetic interaction data from different tasks that require diverse strategies, PAPRIKA teaches models to explore and adapt their behavior on a new task based on environment feedback in-context without more gradient updates. Experimental results show that models fine-tuned with PAPRIKA can effectively transfer their learned decision-making capabilities to entirely unseen tasks without additional training. Unlike traditional training, our approach's primary bottleneck lies in sampling useful interaction data instead of model updates. To improve sample efficiency, we propose a curriculum learning strategy that prioritizes sampling trajectories from tasks with high learning potential. These results suggest a promising path towards AI systems that can autonomously solve novel sequential decision-making problems that require interactions with the external world. 

**Abstract (ZH)**: 有效探索对于与环境互动的智能系统至关重要，但现有语言模型在需要策略性信息收集的场景中往往表现不佳。本文介绍了PAPRIKA，一种微调方法，使语言模型能够发展出不受特定环境限制的一般决策能力。通过在需要不同策略的多种任务中合成交互数据进行训练，PAPRIKA使模型能够根据环境反馈在新任务中探索并调整其行为，而无需更多梯度更新。实验结果表明，使用PAPRIKA微调的语言模型可以在无需额外训练的情况下有效地将学到的决策能力转移到全新的任务中。与传统训练相比，我们方法的主要瓶颈在于采样有用的交互数据而非模型更新。为了提高样本效率，我们提出了一种课程学习策略，优先从具有高学习潜力的任务中采样轨迹。这些结果指出了一个有前景的方向，即自主解决需要与外部世界互动的新型序列决策问题的AI系统。 

---
# PosterSum: A Multimodal Benchmark for Scientific Poster Summarization 

**Title (ZH)**: PosterSum：一种科学海报总结的多模态基准 

**Authors**: Rohit Saxena, Pasquale Minervini, Frank Keller  

**Link**: [PDF](https://arxiv.org/pdf/2502.17540)  

**Abstract**: Generating accurate and concise textual summaries from multimodal documents is challenging, especially when dealing with visually complex content like scientific posters. We introduce PosterSum, a novel benchmark to advance the development of vision-language models that can understand and summarize scientific posters into research paper abstracts. Our dataset contains 16,305 conference posters paired with their corresponding abstracts as summaries. Each poster is provided in image format and presents diverse visual understanding challenges, such as complex layouts, dense text regions, tables, and figures. We benchmark state-of-the-art Multimodal Large Language Models (MLLMs) on PosterSum and demonstrate that they struggle to accurately interpret and summarize scientific posters. We propose Segment & Summarize, a hierarchical method that outperforms current MLLMs on automated metrics, achieving a 3.14% gain in ROUGE-L. This will serve as a starting point for future research on poster summarization. 

**Abstract (ZH)**: 从多模态文档中生成准确简洁的文本摘要具有挑战性，尤其是在处理像科学海报这样的视觉复杂内容时。我们引入了PosterSum，这是一个新的基准，旨在促进能够理解并总结科学海报为研究论文摘要的视觉-语言模型的发展。我们的数据集包含16,305张会议海报及其对应的摘要作为总结。每个海报以图像格式提供，并包含各种视觉理解挑战，如复杂的布局、密集的文字区域、表格和图表。我们在PosterSum上基准测试最先进的多模态大型语言模型（MLLMs），并展示了它们在准确解释和总结科学海报方面的困难。我们提出了一种分段与总结的方法，该方法在自动化评估指标上优于当前的MLLMs，获得了3.14%的ROUGE-L增益。这将为未来的海报总结研究提供一个起点。 

---
# On the Vulnerability of Concept Erasure in Diffusion Models 

**Title (ZH)**: 扩散模型中概念擦除的脆弱性研究 

**Authors**: Lucas Beerens, Alex D. Richardson, Kaicheng Zhang, Dongdong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.17537)  

**Abstract**: The proliferation of text-to-image diffusion models has raised significant privacy and security concerns, particularly regarding the generation of copyrighted or harmful images. To address these issues, research on machine unlearning has developed various concept erasure methods, which aim to remove the effect of unwanted data through post-hoc training. However, we show these erasure techniques are vulnerable, where images of supposedly erased concepts can still be generated using adversarially crafted prompts. We introduce RECORD, a coordinate-descent-based algorithm that discovers prompts capable of eliciting the generation of erased content. We demonstrate that RECORD significantly beats the attack success rate of current state-of-the-art attack methods. Furthermore, our findings reveal that models subjected to concept erasure are more susceptible to adversarial attacks than previously anticipated, highlighting the urgency for more robust unlearning approaches. We open source all our code at this https URL 

**Abstract (ZH)**: 文本到图像扩散模型的普及引发了显著的隐私和安全关切，特别是关于版权受保护或有害图像的生成。为应对这些问题，关于机器遗忘的研究开发了各种概念擦除方法，旨在通过后训练去除不想要数据的影响。然而，我们揭示这些擦除技术存在漏洞， adversarially 制作的提示仍然能够生成被擦除的概念图像。我们介绍了基于坐标下降的 RECORD 算法，该算法能够发现能够引发被擦除内容生成的提示。我们证明了 RECORD 明显优于当前最先进的攻击方法的成功率。此外，我们的研究发现，遭受概念擦除的模型比预期更易受到对抗性攻击，强调了更稳健的遗忘方法的迫切需求。我们已在以下网址开源了所有代码：this https URL。 

---
# The Lottery LLM Hypothesis, Rethinking What Abilities Should LLM Compression Preserve? 

**Title (ZH)**: 彩票LLM假设：重新思考LLM压缩应保留的能力？ 

**Authors**: Zhenheng Tang, Xiang Liu, Qian Wang, Peijie Dong, Bingsheng He, Xiaowen Chu, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.17535)  

**Abstract**: Motivated by reducing the computational and storage costs of LLMs, model compression and KV cache compression have attracted much attention from researchers. However, current methods predominantly emphasize maintaining the performance of compressed LLMs, as measured by perplexity or simple accuracy on tasks of common sense knowledge QA and basic arithmetic reasoning. In this blog, we present a brief review of recent advancements in LLMs related to retrieval-augmented generation, multi-step reasoning, external tools, and computational expressivity, all of which substantially enhance LLM performance. Then, we propose a lottery LLM hypothesis suggesting that for a given LLM and task, there exists a smaller lottery LLM capable of producing the same performance as the original LLM with the assistance of multi-step reasoning and external tools. Based on the review of current progress in LLMs, we discuss and summarize the essential capabilities that the lottery LLM and KV cache compression must possess, which are currently overlooked in existing methods. 

**Abstract (ZH)**: 受减轻大规模语言模型计算和存储成本的驱动，模型压缩和KV缓存压缩引起了研究人员的广泛关注。然而，当前方法主要侧重于保持压缩后的大规模语言模型的性能，这通常通过困惑度或常识知识问答和基本算术推理任务的简单准确性来衡量。在这篇博文中，我们简要回顾了与检索增强生成、多步推理、外部工具和计算表达性相关的最新大规模语言模型进展，这些进展显著提升了语言模型的性能。然后，我们提出了一种彩票语言模型假说，认为对于给定的语言模型和任务，存在一个更小的彩票语言模型，在多步推理和外部工具的帮助下，能够产生与原始语言模型相同的效果。基于当前大规模语言模型的发展，我们讨论并总结了彩票语言模型和KV缓存压缩目前被现有方法所忽视的必要能力。 

---
# From Euler to AI: Unifying Formulas for Mathematical Constants 

**Title (ZH)**: 从欧拉到AI：数学常数的统一致公式 

**Authors**: Tomer Raz, Michael Shalyt, Elyasheev Leibtag, Rotem Kalisch, Yaron Hadad, Ido Kaminer  

**Link**: [PDF](https://arxiv.org/pdf/2502.17533)  

**Abstract**: The constant $\pi$ has fascinated scholars for centuries, inspiring the derivation of countless formulas rooted in profound mathematical insight. This abundance of formulas raises a question: Are they interconnected, and can a unifying structure explain their relationships?
We propose a systematic methodology for discovering and proving formula equivalences, leveraging modern large language models, large-scale data processing, and novel mathematical algorithms. Analyzing 457,145 arXiv papers, over a third of the validated formulas for $\pi$ were proven to be derivable from a single mathematical object - including formulas by Euler, Gauss, Lord Brouncker, and newer ones from algorithmic discoveries by the Ramanujan Machine.
Our approach extends to other constants, such as $e$, $\zeta(3)$, and Catalan's constant, proving its broad applicability. This work represents a step toward the automatic unification of mathematical knowledge, laying a foundation for AI-driven discoveries of connections across scientific domains. 

**Abstract (ZH)**: 圆周率π长久以来一直吸引着学者们的关注，激发了无数蕴含深刻数学洞察的公式推导。这些公式的丰富多样性引发了一个问题：它们之间是否存在联系，是否有一种统一的结构可以解释这些关系？
我们提出了一种系统性方法来发现和证明公式等价性，利用现代大规模语言模型、大规模数据处理以及新型数学算法。分析了457,145篇arXiv论文，超过三分之一的已验证π公式都可以追溯到单一的数学对象——包括欧拉、高斯、布朗克爵士以及拉马努詹机器中新发现算法给出的公式。
我们的方法还可以推广到其他常数，如e、ζ(3)和卡塔兰常数，证明了其广泛的应用性。这项工作朝着自动统一数学知识的方向迈进了一步，为基于AI的跨学科领域发现联系奠定了基础。 

---
# Laplace-Beltrami Operator for Gaussian Splatting 

**Title (ZH)**: 拉普拉斯-贝尔特拉米算子在高斯点云计算中的应用 

**Authors**: Hongyu Zhou, Zorah Lähner  

**Link**: [PDF](https://arxiv.org/pdf/2502.17531)  

**Abstract**: With the rising popularity of 3D Gaussian splatting and the expanse of applications from rendering to 3D reconstruction, there comes also a need for geometry processing applications directly on this new representation. While considering the centers of Gaussians as a point cloud or meshing them is an option that allows to apply existing algorithms, this might ignore information present in the data or be unnecessarily expensive. Additionally, Gaussian splatting tends to contain a large number of outliers which do not affect the rendering quality but need to be handled correctly in order not to produce noisy results in geometry processing applications. In this work, we propose a formulation to compute the Laplace-Beltrami operator, a widely used tool in geometry processing, directly on Gaussian splatting using the Mahalanobis distance. While conceptually similar to a point cloud Laplacian, our experiments show superior accuracy on the point clouds encoded in the Gaussian splatting centers and, additionally, the operator can be used to evaluate the quality of the output during optimization. 

**Abstract (ZH)**: 随着3D高斯簇集的 popularity 上升及其在渲染到 3D 重建等应用领域的扩展，对于这种新表示形式直接进行几何处理的应用需求也随之增加。虽然可以将高斯簇的中心视为点云或将它们网格化以便应用现有算法，但这可能会忽略数据中包含的信息，或者导致不必要的高价。此外，高斯簇集通常包含大量不影响渲染质量但需要在几何处理应用中正确处理的离群值。在本工作中，我们提出了一种直接在高斯簇集上计算拉普拉斯-贝尔特拉米算子的公式，利用马哈拉诺比斯距离。尽管从概念上类似于点云拉普拉斯算子，我们的实验表明，这种方法在编码于高斯簇集中点云上的准确度更优，此外，该算子还可以在优化过程中评估输出质量。 

---
# Perceptual Noise-Masking with Music through Deep Spectral Envelope Shaping 

**Title (ZH)**: 通过深度频谱包络整形实现音乐掩蔽感知噪声 

**Authors**: Clémentine Berger, Roland Badeau, Slim Essid  

**Link**: [PDF](https://arxiv.org/pdf/2502.17527)  

**Abstract**: People often listen to music in noisy environments, seeking to isolate themselves from ambient sounds. Indeed, a music signal can mask some of the noise's frequency components due to the effect of simultaneous masking. In this article, we propose a neural network based on a psychoacoustic masking model, designed to enhance the music's ability to mask ambient noise by reshaping its spectral envelope with predicted filter frequency responses. The model is trained with a perceptual loss function that balances two constraints: effectively masking the noise while preserving the original music mix and the user's chosen listening level. We evaluate our approach on simulated data replicating a user's experience of listening to music with headphones in a noisy environment. The results, based on defined objective metrics, demonstrate that our system improves the state of the art. 

**Abstract (ZH)**: 人们通常在嘈杂环境中听音乐，试图隔离环境噪音。实际上，音乐信号可以通过同时掩蔽效应掩盖部分噪声的频率成分。本文提出了一种基于心理声学掩蔽模型的神经网络，旨在通过预测滤波器频率响应重塑音乐的频谱包络，增强其掩蔽环境噪声的能力。该模型通过感知损失函数进行训练，平衡两个约束：有效掩蔽噪声同时保持原始音乐混音和用户选定的聆听水平。我们在模拟用户在嘈杂环境中通过耳机听音乐的经验数据上评估了该方法。基于定义的客观指标，结果表明我们的系统提升了现有技术的水平。 

---
# Multimodal Bearing Fault Classification Under Variable Conditions: A 1D CNN with Transfer Learning 

**Title (ZH)**: 基于变工况下的多模态轴承故障分类：迁移学习的1D CNN方法 

**Authors**: Tasfiq E. Alam, Md Manjurul Ahsan, Shivakumar Raman  

**Link**: [PDF](https://arxiv.org/pdf/2502.17524)  

**Abstract**: Bearings play an integral role in ensuring the reliability and efficiency of rotating machinery - reducing friction and handling critical loads. Bearing failures that constitute up to 90% of mechanical faults highlight the imperative need for reliable condition monitoring and fault detection. This study proposes a multimodal bearing fault classification approach that relies on vibration and motor phase current signals within a one-dimensional convolutional neural network (1D CNN) framework. The method fuses features from multiple signals to enhance the accuracy of fault detection. Under the baseline condition (1,500 rpm, 0.7 Nm load torque, and 1,000 N radial force), the model reaches an accuracy of 96% with addition of L2 regularization. This represents a notable improvement of 2% compared to the non-regularized model. In addition, the model demonstrates robust performance across three distinct operating conditions by employing transfer learning (TL) strategies. Among the tested TL variants, the approach that preserves parameters up to the first max-pool layer and then adjusts subsequent layers achieves the highest performance. While this approach attains excellent accuracy across varied conditions, it requires more computational time due to its greater number of trainable parameters. To address resource constraints, less computationally intensive models offer feasible trade-offs, albeit at a slight accuracy cost. Overall, this multimodal 1D CNN framework with late fusion and TL strategies lays a foundation for more accurate, adaptable, and efficient bearing fault classification in industrial environments with variable operating conditions. 

**Abstract (ZH)**: 轴承在确保旋转机械的可靠性和效率中发挥着重要作用，通过降低摩擦和承担关键载荷。轴承故障占机械故障的90%左右，凸显了可靠的状态监测和故障检测的必要性。本文提出了一种基于振动和电机相电流信号的一维卷积神经网络（1D CNN）框架下的多模式轴承故障分类方法，通过融合多信号特征以提高故障检测的准确性。在基线条件下（1,500 rpm，0.7 Nm负载扭矩和1,000 N径向力），模型在添加L2正则化后达到96%的准确率，相比非正则化模型提高了2%。此外，通过使用迁移学习（TL）策略，该模型在三种不同的运行条件下均表现出稳健性能。在测试的TL变体中，保留到第一个最大池化层的参数并调整后续层的方法表现最佳。尽管该方法在不同条件下的准确率很高，但由于参数更多，所需的计算时间也更长。为解决资源限制，计算量更轻的模型是可行的替代方案，尽管会略微牺牲准确率。总体而言，该多模式1D CNN框架结合了晚期融合和TL策略，为具有可变运行条件的工业环境中更准确、适应性和高效轴承故障分类奠定了基础。 

---
# Spectral Theory for Edge Pruning in Asynchronous Recurrent Graph Neural Networks 

**Title (ZH)**: 异步循环图神经网络中的边缘剪枝频谱理论 

**Authors**: Nicolas Bessone  

**Link**: [PDF](https://arxiv.org/pdf/2502.17522)  

**Abstract**: Graph Neural Networks (GNNs) have emerged as a powerful tool for learning on graph-structured data, finding applications in numerous domains including social network analysis and molecular biology. Within this broad category, Asynchronous Recurrent Graph Neural Networks (ARGNNs) stand out for their ability to capture complex dependencies in dynamic graphs, resembling living organisms' intricate and adaptive nature. However, their complexity often leads to large and computationally expensive models. Therefore, pruning unnecessary edges becomes crucial for enhancing efficiency without significantly compromising performance. This paper presents a dynamic pruning method based on graph spectral theory, leveraging the imaginary component of the eigenvalues of the network graph's Laplacian. 

**Abstract (ZH)**: 基于图谱理论的异步递归图神经网络动态剪枝方法（利用网络图拉普拉斯矩阵特征值的虚部） 

---
# Recent Advances in Large Langauge Model Benchmarks against Data Contamination: From Static to Dynamic Evaluation 

**Title (ZH)**: 大数据污染背景下大型语言模型基准测试的 recent 进展：从静态评价到动态评价 

**Authors**: Simin Chen, Yiming Chen, Zexin Li, Yifan Jiang, Zhongwei Wan, Yixin He, Dezhi Ran, Tianle Gu, Haizhou Li, Tao Xie, Baishakhi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2502.17521)  

**Abstract**: Data contamination has received increasing attention in the era of large language models (LLMs) due to their reliance on vast Internet-derived training corpora. To mitigate the risk of potential data contamination, LLM benchmarking has undergone a transformation from static to dynamic benchmarking. In this work, we conduct an in-depth analysis of existing static to dynamic benchmarking methods aimed at reducing data contamination risks. We first examine methods that enhance static benchmarks and identify their inherent limitations. We then highlight a critical gap-the lack of standardized criteria for evaluating dynamic benchmarks. Based on this observation, we propose a series of optimal design principles for dynamic benchmarking and analyze the limitations of existing dynamic benchmarks. This survey provides a concise yet comprehensive overview of recent advancements in data contamination research, offering valuable insights and a clear guide for future research efforts. We maintain a GitHub repository to continuously collect both static and dynamic benchmarking methods for LLMs. The repository can be found at this link. 

**Abstract (ZH)**: 大数据污染在大规模语言模型时代日益受到关注，因此动态基准测试成为减轻数据污染风险的重要途径。本文深入分析现有从静态到动态基准测试的方法，旨在减少数据污染风险。我们首先探讨了提升静态基准测试的方法，并指出了其固有局限性。然后，我们指出一个关键不足——缺乏标准化的动态基准测试评估标准。基于此观察，我们提出了动态基准测试的设计原则，并分析了现有动态基准测试的局限性。本文提供了一个简洁而全面的数据污染研究进展概览，为未来研究提供了宝贵的见解和清晰的指导。我们维护一个GitHub仓库，持续收集大规模语言模型的静态和动态基准测试方法。仓库链接请见此网址。 

---
# Ensemble RL through Classifier Models: Enhancing Risk-Return Trade-offs in Trading Strategies 

**Title (ZH)**: 通过分类器模型的集成RL：优化交易策略中的风险收益权衡 

**Authors**: Zheli Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2502.17518)  

**Abstract**: This paper presents a comprehensive study on the use of ensemble Reinforcement Learning (RL) models in financial trading strategies, leveraging classifier models to enhance performance. By combining RL algorithms such as A2C, PPO, and SAC with traditional classifiers like Support Vector Machines (SVM), Decision Trees, and Logistic Regression, we investigate how different classifier groups can be integrated to improve risk-return trade-offs. The study evaluates the effectiveness of various ensemble methods, comparing them with individual RL models across key financial metrics, including Cumulative Returns, Sharpe Ratios (SR), Calmar Ratios, and Maximum Drawdown (MDD). Our results demonstrate that ensemble methods consistently outperform base models in terms of risk-adjusted returns, providing better management of drawdowns and overall stability. However, we identify the sensitivity of ensemble performance to the choice of variance threshold {\tau}, highlighting the importance of dynamic {\tau} adjustment to achieve optimal performance. This study emphasizes the value of combining RL with classifiers for adaptive decision-making, with implications for financial trading, robotics, and other dynamic environments. 

**Abstract (ZH)**: 本文全面研究了集成强化学习（RL）模型在金融交易策略中的应用，通过结合A2C、PPO、SAC等RL算法与SVM、决策树、逻辑回归等传统分类器，探讨不同分类器组合如何改善风险收益权衡。研究评估了各种集成方法的有效性，并与单一RL模型在累计回报、夏普比率、卡马比率和最大回撤等关键金融指标上进行比较。结果表明，集成方法在风险调整回报方面始终优于基础模型，提供了更好的回撤管理和整体稳定性。然而，研究指出集成性能对方差阈值τ的选择高度敏感，强调了动态调整τ的重要性以实现最佳性能。本文强调了将RL与分类器结合用于自适应决策的价值，对金融交易、机器人技术和其他动态环境具有重要意义。 

---
# Attention-based UAV Trajectory Optimization for Wireless Power Transfer-assisted IoT Systems 

**Title (ZH)**: 基于注意力机制的无人机轨迹优化以支持无线能量传输辅助物联网系统 

**Authors**: Li Dong, Feibo Jiang, Yubo Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.17517)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) in Wireless Power Transfer (WPT)-assisted Internet of Things (IoT) systems face the following challenges: limited resources and suboptimal trajectory planning. Reinforcement learning-based trajectory planning schemes face issues of low search efficiency and learning instability when optimizing large-scale systems. To address these issues, we present an Attention-based UAV Trajectory Optimization (AUTO) framework based on the graph transformer, which consists of an Attention Trajectory Optimization Model (ATOM) and a Trajectory lEarNing Method based on Actor-critic (TENMA). In ATOM, a graph encoder is used to calculate the self-attention characteristics of all IoTDs, and a trajectory decoder is developed to optimize the number and trajectories of UAVs. TENMA then trains the ATOM using an improved Actor-Critic method, in which the real reward of the system is applied as the baseline to reduce variances in the critic network. This method is suitable for high-quality and large-scale multi-UAV trajectory planning. Finally, we develop numerous experiments, including a hardware experiment in the field case, to verify the feasibility and efficiency of the AUTO framework. 

**Abstract (ZH)**: 基于图变换器的注意力机制无人机轨迹优化（AUTO）框架：面向无线能量传输辅助物联网系统的注意力轨迹优化模型（ATOM）与基于 actor-critic 的轨迹学习方法（TENMA） 

---
# A Survey on Mechanistic Interpretability for Multi-Modal Foundation Models 

**Title (ZH)**: 多模态基础模型的机理可解释性综述 

**Authors**: Zihao Lin, Samyadeep Basu, Mohammad Beigi, Varun Manjunatha, Ryan A. Rossi, Zichao Wang, Yufan Zhou, Sriram Balasubramanian, Arman Zarei, Keivan Rezaei, Ying Shen, Barry Menglong Yao, Zhiyang Xu, Qin Liu, Yuxiang Zhang, Yan Sun, Shilong Liu, Li Shen, Hongxuan Li, Soheil Feizi, Lifu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17516)  

**Abstract**: The rise of foundation models has transformed machine learning research, prompting efforts to uncover their inner workings and develop more efficient and reliable applications for better control. While significant progress has been made in interpreting Large Language Models (LLMs), multimodal foundation models (MMFMs) - such as contrastive vision-language models, generative vision-language models, and text-to-image models - pose unique interpretability challenges beyond unimodal frameworks. Despite initial studies, a substantial gap remains between the interpretability of LLMs and MMFMs. This survey explores two key aspects: (1) the adaptation of LLM interpretability methods to multimodal models and (2) understanding the mechanistic differences between unimodal language models and crossmodal systems. By systematically reviewing current MMFM analysis techniques, we propose a structured taxonomy of interpretability methods, compare insights across unimodal and multimodal architectures, and highlight critical research gaps. 

**Abstract (ZH)**: 基础模型的兴起已改变了机器学习研究，促使人们努力揭示其内在工作机制，并开发更高效和可靠的多模态应用以实现更好控制。尽管在解释大规模语言模型方面取得了显著进展，但对比视觉语言模型、生成式视觉语言模型和文本到图像模型等多模态基础模型（MMFMs）提出了超越单模态框架的独特可解释性挑战。尽管初步研究已经开展，大规模语言模型和多模态基础模型之间的可解释性差距仍然较大。本综述探讨了两个关键方面：（1）将大规模语言模型的解释方法适应多模态模型；（2）理解单模态语言模型与跨模态系统之间的机制差异。通过系统性地回顾当前的多模态基础模型分析技术，我们提出了一种结构化的解释方法分类体系，比较了单模态和多模态架构之间的洞见，并突出了关键的研究空白。 

---
# Towards User-level Private Reinforcement Learning with Human Feedback 

**Title (ZH)**: 面向用户的私有强化学习与人类反馈 

**Authors**: Jiaming Zhang, Mingxi Lei, Meng Ding, Mengdi Li, Zihang Xiang, Difei Xu, Jinhui Xu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17515)  

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) has emerged as an influential technique, enabling the alignment of large language models (LLMs) with human preferences. Despite the promising potential of RLHF, how to protect user preference privacy has become a crucial issue. Most previous work has focused on using differential privacy (DP) to protect the privacy of individual data. However, they have concentrated primarily on item-level privacy protection and have unsatisfactory performance for user-level privacy, which is more common in RLHF. This study proposes a novel framework, AUP-RLHF, which integrates user-level label DP into RLHF. We first show that the classical random response algorithm, which achieves an acceptable performance in item-level privacy, leads to suboptimal utility when in the user-level settings. We then establish a lower bound for the user-level label DP-RLHF and develop the AUP-RLHF algorithm, which guarantees $(\varepsilon, \delta)$ user-level privacy and achieves an improved estimation error. Experimental results show that AUP-RLHF outperforms existing baseline methods in sentiment generation and summarization tasks, achieving a better privacy-utility trade-off. 

**Abstract (ZH)**: 基于人类反馈的强化学习（RLHF）与用户偏好隐私保护：AUP-RLHF框架 

---
# SAE-V: Interpreting Multimodal Models for Enhanced Alignment 

**Title (ZH)**: SAE-V: 多模态模型的解释以提高对齐效果 

**Authors**: Hantao Lou, Changye Li, Jiaming Ji, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17514)  

**Abstract**: With the integration of image modality, the semantic space of multimodal large language models (MLLMs) is more complex than text-only models, making their interpretability more challenging and their alignment less stable, particularly susceptible to low-quality data, which can lead to inconsistencies between modalities, hallucinations, and biased outputs. As a result, developing interpretability methods for MLLMs is crucial for improving alignment quality and efficiency. In text-only LLMs, Sparse Autoencoders (SAEs) have gained attention for their ability to interpret latent representations. However, extending SAEs to multimodal settings presents new challenges due to modality fusion and the difficulty of isolating cross-modal representations. To address these challenges, we introduce SAE-V, a mechanistic interpretability framework that extends the SAE paradigm to MLLMs. By identifying and analyzing interpretable features along with their corresponding data, SAE-V enables fine-grained interpretation of both model behavior and data quality, facilitating a deeper understanding of cross-modal interactions and alignment dynamics. Moreover, by utilizing cross-modal feature weighting, SAE-V provides an intrinsic data filtering mechanism to enhance model alignment without requiring additional models. Specifically, when applied to the alignment process of MLLMs, SAE-V-based data filtering methods could achieve more than 110% performance with less than 50% data. Our results highlight SAE-V's ability to enhance interpretability and alignment in MLLMs, providing insights into their internal mechanisms. 

**Abstract (ZH)**: 基于图像的模态整合使多模态大规模语言模型（MLLMs）的语义空间更加复杂，这使得解释其行为更加具有挑战性，模型对齐也更加不稳定，特别容易受到低质量数据的影响，从而导致模态间不一致、幻觉和有偏输出。因此，开发针对MLLMs的解释性方法对于提高对齐质量和效率至关重要。在仅文本的大规模语言模型（text-only LLMs）中，稀疏自编码器（SAEs）由于其能够解释潜在表示的能力而备受关注。然而，将SAEs扩展到多模态设置中带来了新的挑战，这些挑战主要来自于模态融合和跨模态表示难以隔离。为了解决这些挑战，我们提出了SAE-V，这是一种基于机理的解释框架，将SAE范式扩展到MLLMs中。通过识别和分析可解释的特征及其对应的多模态数据，SAE-V能够对模型行为和数据质量进行精细解释，有助于更深入地理解跨模态交互和对齐动力学。此外，通过利用跨模态特征加权，SAE-V提供了一个内在的数据过滤机制，无需额外模型即可增强模型对齐。具体来说，在MLLMs的对齐过程中，基于SAE-V的数据过滤方法可以在少于50%数据的情况下达到超过110%的性能提升。我们的结果突显了SAE-V增强MLLMs解释性和对齐能力的能力，并提供了其内部机制的见解。 

---
# Int2Int: a framework for mathematics with transformers 

**Title (ZH)**: Int2Int：一种基于变换器的数学框架 

**Authors**: François Charton  

**Link**: [PDF](https://arxiv.org/pdf/2502.17513)  

**Abstract**: This paper documents Int2Int, an open source code base for using transformers on problems of mathematical research, with a focus on number theory and other problems involving integers. Int2Int is a complete PyTorch implementation of a transformer architecture, together with training and evaluation loops, and classes and functions to represent, generate and decode common mathematical objects. Ancillary code for data preparation, and Jupyter Notebooks for visualizing experimental results are also provided. This document presents the main features of Int2Int, serves as its user manual, and provides guidelines on how to extend it. Int2Int is released under the MIT licence, at this https URL. 

**Abstract (ZH)**: 本论文记录了Int2Int，一个用于数学研究问题的开源代码库，重点关注数论及其他涉及整数的问题。Int2Int是基于PyTorch的变压器架构的完整实现，并包含训练和评估循环，以及表示、生成和解码常见数学对象的类和函数。还提供了数据准备辅助代码和用于可视化实验结果的Jupyter Notebook。本文档介绍了Int2Int的主要功能，作为其用户手册，并提供了扩展它的指导。Int2Int在MIT许可证下发布，详见此处：https://github.com/int2int/int2int 

---
# Recurrent Knowledge Identification and Fusion for Language Model Continual Learning 

**Title (ZH)**: 循环知识识别与融合在语言模型持续学习中的应用 

**Authors**: Yujie Feng, Xujia Wang, Zexin Lu, Shenghong Fu, Guangyuan Shi, Yongxin Xu, Yasha Wang, Philip S. Yu, Xu Chu, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17510)  

**Abstract**: Continual learning (CL) is crucial for deploying large language models (LLMs) in dynamic real-world environments without costly retraining. While recent model ensemble and model merging methods guided by parameter importance have gained popularity, they often struggle to balance knowledge transfer and forgetting, mainly due to the reliance on static importance estimates during sequential training. In this paper, we present Recurrent-KIF, a novel CL framework for Recurrent Knowledge Identification and Fusion, which enables dynamic estimation of parameter importance distributions to enhance knowledge transfer. Inspired by human continual learning, Recurrent-KIF employs an inner loop that rapidly adapts to new tasks while identifying important parameters, coupled with an outer loop that globally manages the fusion of new and historical knowledge through redundant knowledge pruning and key knowledge merging. These inner-outer loops iteratively perform multiple rounds of fusion, allowing Recurrent-KIF to leverage intermediate training information and adaptively adjust fusion strategies based on evolving importance distributions. Extensive experiments on two CL benchmarks with various model sizes (from 770M to 13B) demonstrate that Recurrent-KIF effectively mitigates catastrophic forgetting and enhances knowledge transfer. 

**Abstract (ZH)**: 持续学习（CL）对于在动态现实环境中部署大规模语言模型（LLMs）至关重要，无需高昂的重新训练成本。尽管基于参数重要性的模型集成和模型融合方法近年来 popularity，它们在平衡知识传递和遗忘方面往往遇到困难，主要归因于顺序训练过程中依赖静态的重要性估计。在本文中，我们提出了循环-KIF，一种新颖的循环知识识别与融合（Recurrent Knowledge Identification and Fusion）持续学习框架，能够动态估计参数重要性分布以增强知识传递。受人类持续学习启发，循环-KIF 采用一个内部循环迅速适应新任务并识别重要参数，结合一个外部循环通过冗余知识剪枝和关键知识合并来全局管理新旧知识的融合。这些内部-外部循环迭代进行多次融合，使循环-KIF 能够利用中间训练信息并根据演化的的重要性分布自适应调整融合策略。在两个不同规模模型（从770M到13B）的持续学习基准实验中，广泛的实验结果表明循环-KIF 有效地缓解了灾难性遗忘并增强了知识传递。 

---
# C-3DPO: Constrained Controlled Classification for Direct Preference Optimization 

**Title (ZH)**: C-3DPO：受约束的控制分类以实现直接偏好优化 

**Authors**: Kavosh Asadi, Julien Han, Xingzi Xu, Dominique Perrault-Joncas, Shoham Sabach, Karim Bouyarmane, Mohammad Ghavamzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2502.17507)  

**Abstract**: Direct preference optimization (DPO)-style algorithms have emerged as a promising approach for solving the alignment problem in AI. We present a novel perspective that formulates these algorithms as implicit classification algorithms. This classification framework enables us to recover many variants of DPO-style algorithms by choosing appropriate classification labels and loss functions. We then leverage this classification framework to demonstrate that the underlying problem solved in these algorithms is under-specified, making them susceptible to probability collapse of the winner-loser responses. We address this by proposing a set of constraints designed to control the movement of probability mass between the winner and loser in the reference and target policies. Our resulting algorithm, which we call Constrained Controlled Classification DPO (\texttt{C-3DPO}), has a meaningful RLHF interpretation. By hedging against probability collapse, \texttt{C-3DPO} provides practical improvements over vanilla \texttt{DPO} when aligning several large language models using standard preference datasets. 

**Abstract (ZH)**: 直接偏好优化（DPO）风格的算法已成为解决人工智能对齐问题的一种有 promise 的方法。我们提出了一种新的视角，将这些算法形式化为隐式分类算法。该分类框架允许我们通过选择合适的分类标签和损失函数来恢复许多种 DPO 风格的算法。然后，借助这一分类框架，我们证明这些算法解决的基础问题存在定义不足，使其容易出现赢家输家响应的概率崩溃。为此，我们提出了一系列约束条件，旨在控制参考策略和目标策略中赢家和输家之间概率质量的移动。我们提出的一种新的算法，称为受约束的控制分类 DPO（\texttt{C-3DPO}），具有有意义的自监督人工智能对齐解释。通过抵消概率崩溃风险，\texttt{C-3DPO} 在使用标准偏好数据集对齐多个大型语言模型时提供了实际改进。 

---
# RAG-Enhanced Collaborative LLM Agents for Drug Discovery 

**Title (ZH)**: RAG增强的合作型大语言模型药物发现代理 

**Authors**: Namkyeong Lee, Edward De Brouwer, Ehsan Hajiramezanali, Chanyoung Park, Gabriele Scalia  

**Link**: [PDF](https://arxiv.org/pdf/2502.17506)  

**Abstract**: Recent advances in large language models (LLMs) have shown great potential to accelerate drug discovery. However, the specialized nature of biochemical data often necessitates costly domain-specific fine-tuning, posing critical challenges. First, it hinders the application of more flexible general-purpose LLMs in cutting-edge drug discovery tasks. More importantly, it impedes the rapid integration of the vast amounts of scientific data continuously generated through experiments and research. To investigate these challenges, we propose CLADD, a retrieval-augmented generation (RAG)-empowered agentic system tailored to drug discovery tasks. Through the collaboration of multiple LLM agents, CLADD dynamically retrieves information from biomedical knowledge bases, contextualizes query molecules, and integrates relevant evidence to generate responses -- all without the need for domain-specific fine-tuning. Crucially, we tackle key obstacles in applying RAG workflows to biochemical data, including data heterogeneity, ambiguity, and multi-source integration. We demonstrate the flexibility and effectiveness of this framework across a variety of drug discovery tasks, showing that it outperforms general-purpose and domain-specific LLMs as well as traditional deep learning approaches. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）在加速药物发现中的应用潜力已经显现。然而，生物化学数据的专业性质往往需要成本高昂的领域特定微调，提出了关键挑战。首先，这阻碍了更灵活的通用LLMs在前沿药物发现任务中的应用。更重要的是，这阻碍了快速整合通过实验和研究不断生成的大量科学数据。为了研究这些挑战，我们提出了CLADD，这是一种通过检索增强生成（RAG）赋能的针对药物发现任务的自主系统。通过多个LLM代理的合作，CLADD动态检索生物医学知识库中的信息，上下文化查询分子，并整合相关证据生成响应，而无需领域特定微调。至关重要的是，我们解决了将RAG工作流应用于生物化学数据的关键障碍，包括数据异质性、模糊性和多源整合。我们展示了该框架在各种药物发现任务中的灵活性和有效性，表明其性能优于通用和领域特定的大规模语言模型以及传统的深度学习方法。 

---
# Inverse Surrogate Model of a Soft X-Ray Spectrometer using Domain Adaptation 

**Title (ZH)**: 软X射线谱仪的域适应逆代理模型 

**Authors**: Enrico Ahlers, Peter Feuer-Forson, Gregor Hartmann, Rolf Mitzner, Peter Baumgärtel, Jens Viefhaus  

**Link**: [PDF](https://arxiv.org/pdf/2502.17505)  

**Abstract**: In this study, we present a method to create a robust inverse surrogate model for a soft X-ray spectrometer. During a beamtime at an electron storage ring, such as BESSY II, instrumentation and beamlines are required to be correctly aligned and calibrated for optimal experimental conditions. In order to automate these processes, machine learning methods can be developed and implemented, but in many cases these methods require the use of an inverse model which maps the output of the experiment, such as a detector image, to the parameters of the device. Due to limited experimental data, such models are often trained with simulated data, which creates the challenge of compensating for the inherent differences between simulation and experiment. In order to close this gap, we demonstrate the application of data augmentation and adversarial domain adaptation techniques, with which we can predict absolute coordinates for the automated alignment of our spectrometer. Bridging the simulation-experiment gap with minimal real-world data opens new avenues for automated experimentation using machine learning in scientific instrumentation. 

**Abstract (ZH)**: 在本研究中，我们提出了一种用于软X射线能谱仪的稳健逆代理模型的方法。在电子储存环（如BESSY II）的束流时间内，需要正确对准和校准仪器和光束线以获得最佳实验条件。为了自动化这些过程，可以开发和实施机器学习方法，但在很多情况下，这些方法需要使用逆模型将实验输出（如探测器图像）映射到设备参数。由于实验数据有限，通常需要使用模拟数据进行训练，这造成了模拟与实验之间固有差异的补偿挑战。为了弥合这一差距，我们展示了数据扩增和对抗域适应技术的应用，这些技术可以预测绝对坐标以实现对我们的能谱仪的自动化对准。使用少量实际数据弥合模拟与实验之间的差距为科学仪器中机器学习驱动的自动化实验开创了新的途径。 

---
# Protein Large Language Models: A Comprehensive Survey 

**Title (ZH)**: 蛋白质大型语言模型：综合调查 

**Authors**: Yijia Xiao, Wanjia Zhao, Junkai Zhang, Yiqiao Jin, Han Zhang, Zhicheng Ren, Renliang Sun, Haixin Wang, Guancheng Wan, Pan Lu, Xiao Luo, Yu Zhang, James Zou, Yizhou Sun, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17504)  

**Abstract**: Protein-specific large language models (Protein LLMs) are revolutionizing protein science by enabling more efficient protein structure prediction, function annotation, and design. While existing surveys focus on specific aspects or applications, this work provides the first comprehensive overview of Protein LLMs, covering their architectures, training datasets, evaluation metrics, and diverse applications. Through a systematic analysis of over 100 articles, we propose a structured taxonomy of state-of-the-art Protein LLMs, analyze how they leverage large-scale protein sequence data for improved accuracy, and explore their potential in advancing protein engineering and biomedical research. Additionally, we discuss key challenges and future directions, positioning Protein LLMs as essential tools for scientific discovery in protein science. Resources are maintained at this https URL. 

**Abstract (ZH)**: 蛋白质特异性大规模语言模型（Protein LLMs）正通过提高蛋白质结构预测、功能注释和设计的效率，革新蛋白质科学。尽管现有综述侧重于特定方面或应用，本工作首次全面概述了Protein LLMs，涵盖其架构、训练数据集、评估指标和多样化的应用。通过对超过100篇文章的系统分析，我们提出了一种结构化的State-of-the-Art Protein LLMs分类体系，分析了它们如何利用大规模蛋白质序列数据以提高准确性，并探讨了它们在推进蛋白质工程和生物医药研究中的潜力。此外，我们讨论了关键挑战和未来方向，将Protein LLMs定位为蛋白质科学研究中必不可少的发现工具。相关资源维护在该网址：https://。 

---
# Doctor-in-the-Loop: An Explainable, Multi-View Deep Learning Framework for Predicting Pathological Response in Non-Small Cell Lung Cancer 

**Title (ZH)**: 医生参与循环的解释性多视图深度学习框架：非小细胞肺癌病理反应预测 

**Authors**: Alice Natalina Caragliano, Filippo Ruffini, Carlo Greco, Edy Ippolito, Michele Fiore, Claudia Tacconi, Lorenzo Nibid, Giuseppe Perrone, Sara Ramella, Paolo Soda, Valerio Guarrasi  

**Link**: [PDF](https://arxiv.org/pdf/2502.17503)  

**Abstract**: Non-small cell lung cancer (NSCLC) remains a major global health challenge, with high post-surgical recurrence rates underscoring the need for accurate pathological response predictions to guide personalized treatments. Although artificial intelligence models show promise in this domain, their clinical adoption is limited by the lack of medically grounded guidance during training, often resulting in non-explainable intrinsic predictions. To address this, we propose Doctor-in-the-Loop, a novel framework that integrates expert-driven domain knowledge with explainable artificial intelligence techniques, directing the model toward clinically relevant anatomical regions and improving both interpretability and trustworthiness. Our approach employs a gradual multi-view strategy, progressively refining the model's focus from broad contextual features to finer, lesion-specific details. By incorporating domain insights at every stage, we enhance predictive accuracy while ensuring that the model's decision-making process aligns more closely with clinical reasoning. Evaluated on a dataset of NSCLC patients, Doctor-in-the-Loop delivers promising predictive performance and provides transparent, justifiable outputs, representing a significant step toward clinically explainable artificial intelligence in oncology. 

**Abstract (ZH)**: 非小细胞肺癌（NSCLC）仍是一项重大的全球健康挑战，术后高复发率突显了需要准确的病理反应预测以指导个性化治疗的必要性。尽管人工智能模型在这一领域显示出潜力，但由于训练过程中缺乏医学依据的指导，其临床应用受到限制，常常导致不可解释的内在预测。为解决这一问题，我们提出了一种名为“医生在环”的新型框架，该框架将专家驱动的领域知识与可解释的人工智能技术相结合，使模型更关注临床相关解剖区域，并提高模型的可解释性和可信度。我们的方法采用逐步多视图策略，逐步细化模型对从宏观上下文特征到更精细、病灶特定细节的聚焦。通过在每个阶段都融入领域见解，我们增强了预测准确性，同时确保模型的决策过程与临床推理更一致。在NSCLC患者的数据库上进行评估，“医生在环”提供了令人鼓舞的预测性能，并提供了透明且可解释的输出，代表了在肿瘤学中实现临床可解释的人工智能的重要一步。 

---
# CoKV: Optimizing KV Cache Allocation via Cooperative Game 

**Title (ZH)**: CoKV：通过合作博弈优化KV缓存分配 

**Authors**: Qiheng Sun, Hongwei Zhang, Haocheng Xia, Jiayao Zhang, Jinfei Liu, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2502.17501)  

**Abstract**: Large language models (LLMs) have achieved remarkable success on various aspects of human life. However, one of the major challenges in deploying these models is the substantial memory consumption required to store key-value pairs (KV), which imposes significant resource demands. Recent research has focused on KV cache budget allocation, with several approaches proposing head-level budget distribution by evaluating the importance of individual attention heads. These methods, however, assess the importance of heads independently, overlooking their cooperative contributions within the model, which may result in a deviation from their true impact on model performance. In light of this limitation, we propose CoKV, a novel method that models the cooperation between heads in model inference as a cooperative game. By evaluating the contribution of each head within the cooperative game, CoKV can allocate the cache budget more effectively. Extensive experiments show that CoKV achieves state-of-the-art performance on the LongBench benchmark using LLama-3-8B-Instruct and Mistral-7B models. 

**Abstract (ZH)**: 大型语言模型（LLMs）在人类生活的多个方面取得了显著成功。然而，部署这些模型的一个主要挑战是存储键值对（KV）所需的大量内存消耗，这带来了显著的资源需求。近期研究聚焦于KV缓存预算分配，提出了一些方法通过评估单个注意力头的重要性来进行头部预算分配。然而，这些方法独立评估每个头的重要性，忽视了它们在模型中的协同贡献，可能导致对其对模型性能真实影响的偏差。为克服这一局限，我们提出了一种名为CoKV的新方法，该方法将模型推理中头部之间的合作视为合作博弈。通过评估每个头在合作博弈中的贡献，CoKV能够更有效地分配缓存预算。大量实验表明，CoKV在LongBench基准上使用LLama-3-8B-Instruct和Mistral-7B模型时实现了最先进的性能。 

---
# Generalized Exponentiated Gradient Algorithms Using the Euler Two-Parameter Logarithm 

**Title (ZH)**: 广义指数梯度算法使用欧拉两参数对数函数 

**Authors**: Andrzej Cichocki  

**Link**: [PDF](https://arxiv.org/pdf/2502.17500)  

**Abstract**: In this paper we propose and investigate a new class of Generalized Exponentiated Gradient (GEG) algorithms using Mirror Descent (MD) approaches, and applying as a regularization function the Bregman divergence with two-parameter deformation of logarithm as a link function. This link function (referred to as the Euler logarithm) is associated with a wide class of generalized entropies. In order to derive novel GEG/MD updates, we estimate generalized exponential function, which closely approximates the inverse of the Euler two-parameter logarithm. The characteristic/shape and properties of the Euler logarithm and its inverse -- deformed exponential functions are tuned by two or even more hyperparameters. By learning these hyperparameters, we can adapt to distribution of training data, and we can adjust them to achieve desired properties of gradient descent algorithms. The concept of generalized entropies and associated deformed logarithms provide deeper insight into novel gradient descent updates.
In literature, there exist nowadays over fifty mathematically well-defined entropic functionals and associated deformed logarithms, so impossible to investigate all of them in one research paper. Therefore, we focus here on a wide-class of trace-form entropies and associated generalized logarithm. We applied the developed algorithms for Online Portfolio Selection (OPLS) in order to improve its performance and robustness. 

**Abstract (ZH)**: 本文提出并研究了一类新的广义指数梯度（GEG）算法，该算法基于镜像下降（MD）方法，并使用带有两参数变形对数作为关联函数的Bregman发散作为正则化函数。该关联函数（称为欧拉对数）与一类广义熵相关。为了导出新的GEG/MD更新，我们估计广义指数函数，该函数紧密逼近欧拉两参数对数的逆函数。欧拉对数及其逆函数——变形指数函数的特点/形状和属性由两个或更多的超参数调节。通过学习这些超参数，可适应训练数据的分布，并可调整它们以实现梯度下降算法所需的特性。广义熵和相关的变形对数为新型梯度下降更新提供了更深层次的理解。

文献中目前存在超过五十种数学上严格定义的熵泛函及相关变形对数，因此在一篇研究论文中无法全面调查它们。因此，本文集中在一类迹形式熵及其相关的广义对数上。我们开发的算法应用于在线组合选择（OPLS）以提高其性能和鲁棒性。 

---
# Accuracy of Wearable ECG Parameter Calculation Method for Long QT and First-Degree A-V Block Detection: A Multi-Center Real-World Study with External Validations Compared to Standard ECG Machines and Cardiologist Assessments 

**Title (ZH)**: 穿戴式ECG参数计算方法用于长QT和一度房室传导阻滞检测的准确性：一项多中心真实世界研究，并与标准ECG机器及心脏病专家评估进行外部验证 

**Authors**: Sumei Fan, Deyun Zhang, Yue Wang, Shijia Geng, Kun Lu, Meng Sang, Weilun Xu, Haixue Wang, Qinghao Zhao, Chuandong Cheng, Peng Wang, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2502.17499)  

**Abstract**: In recent years, wearable devices have revolutionized cardiac monitoring by enabling continuous, non-invasive ECG recording in real-world settings. Despite these advances, the accuracy of ECG parameter calculations (PR interval, QRS interval, QT interval, etc.) from wearables remains to be rigorously validated against conventional ECG machines and expert clinician assessments. In this large-scale, multicenter study, we evaluated FeatureDB, a novel algorithm for automated computation of ECG parameters from wearable single-lead signals Three diverse datasets were employed: the AHMU-FH dataset (n=88,874), the CSE dataset (n=106), and the HeartVoice-ECG-lite dataset (n=369) with annotations provided by two experienced cardiologists. FeatureDB demonstrates a statistically significant correlation with key parameters (PR interval, QRS duration, QT interval, and QTc) calculated by standard ECG machines and annotated by clinical doctors. Bland-Altman analysis confirms a high level of this http URL,FeatureDB exhibited robust diagnostic performance in detecting Long QT syndrome (LQT) and atrioventricular block interval abnormalities (AVBI),with excellent area under the ROC curve (LQT: 0.836, AVBI: 0.861),accuracy (LQT: 0.856, AVBI: 0.845),sensitivity (LQT: 0.815, AVBI: 0.877),and specificity (LQT: 0.856, AVBI: 0.845).This further validates its clinical reliability. These results validate the clinical applicability of FeatureDB for wearable ECG analysis and highlight its potential to bridge the gap between traditional diagnostic methods and emerging wearable this http URL,this study supports integrating wearable ECG devices into large-scale cardiovascular disease management and early intervention strategies,and it highlights the potential of wearable ECG technologies to deliver accurate,clinically relevant cardiac monitoring while advancing broader applications in cardiovascular care. 

**Abstract (ZH)**: 穿戴设备在心电图参数计算中的准确性和临床应用：FeatureDB算法在大型多中心研究中的评估 

---
# Improving Value-based Process Verifier via Structural Prior Injection 

**Title (ZH)**: 基于结构先验注入的价值导向过程验证器改进 

**Authors**: Zetian Sun, Dongfang Li, Baotian Hu, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17498)  

**Abstract**: In the Large Language Model(LLM) reasoning scenario, people often estimate state value via Monte Carlo sampling. Though Monte Carlo estimation is an elegant method with less inductive bias, noise and errors are inevitably introduced due to the limited sampling. To handle the problem, we inject the structural prior into the value representation and transfer the scalar value into the expectation of a pre-defined categorical distribution, representing the noise and errors from a distribution perspective. Specifically, by treating the result of Monte Carlo sampling as a single sample from the prior ground-truth Binomial distribution, we quantify the sampling error as the mismatch between posterior estimated distribution and ground-truth distribution, which is thus optimized via distribution selection optimization. We test the performance of value-based process verifiers on Best-of-N task and Beam search task. Compared with the scalar value representation, we show that reasonable structural prior injection induced by different objective functions or optimization methods can improve the performance of value-based process verifiers for about 1$\sim$2 points at little-to-no cost. We also show that under different structural prior, the verifiers' performances vary greatly despite having the same optimal solution, indicating the importance of reasonable structural prior injection. 

**Abstract (ZH)**: 在大型语言模型(LLM)推理场景中，人们常通过蒙特卡洛采样估计状态值。尽管蒙特卡洛估计方法简洁且具有较少的归纳偏置，但由于采样量有限，噪声和误差不可避免地被引入。为解决这一问题，我们向价值表示中注入结构先验，将标量值转换为预定义分类分布的期望，从分布的角度代表噪声和误差。具体而言，我们将蒙特卡洛采样的结果视为先验真实二项分布的一个样本，量化采样误差为后验估计分布与真实分布之间的不匹配程度，并通过分布选择优化进行优化。我们在Best-of-N任务和Beam搜索任务中测试基于价值的过程验证器的性能。与标量值表示相比，我们显示通过不同目标函数或优化方法引入合理的结构先验可以仅以微小或无成本提高大约1-2个点的验证器性能。同时，我们表明，在不同结构先验下，尽管最优解相同，验证器的性能差异很大，这强调了合理结构先验注入的重要性。 

---
# SpikeRL: A Scalable and Energy-efficient Framework for Deep Spiking Reinforcement Learning 

**Title (ZH)**: SpikeRL：一种可扩展和能效高的深度尖峰强化学习框架 

**Authors**: Tokey Tahmid, Mark Gates, Piotr Luszczek, Catherine D. Schuman  

**Link**: [PDF](https://arxiv.org/pdf/2502.17496)  

**Abstract**: In this era of AI revolution, massive investments in large-scale data-driven AI systems demand high-performance computing, consuming tremendous energy and resources. This trend raises new challenges in optimizing sustainability without sacrificing scalability or performance. Among the energy-efficient alternatives of the traditional Von Neumann architecture, neuromorphic computing and its Spiking Neural Networks (SNNs) are a promising choice due to their inherent energy efficiency. However, in some real-world application scenarios such as complex continuous control tasks, SNNs often lack the performance optimizations that traditional artificial neural networks have. Researchers have addressed this by combining SNNs with Deep Reinforcement Learning (DeepRL), yet scalability remains unexplored. In this paper, we extend our previous work on SpikeRL, which is a scalable and energy efficient framework for DeepRL-based SNNs for continuous control. In our initial implementation of SpikeRL framework, we depended on the population encoding from the Population-coded Spiking Actor Network (PopSAN) method for our SNN model and implemented distributed training with Message Passing Interface (MPI) through mpi4py. Also, further optimizing our model training by using mixed-precision for parameter updates. In our new SpikeRL framework, we have implemented our own DeepRL-SNN component with population encoding, and distributed training with PyTorch Distributed package with NCCL backend while still optimizing with mixed precision training. Our new SpikeRL implementation is 4.26X faster and 2.25X more energy efficient than state-of-the-art DeepRL-SNN methods. Our proposed SpikeRL framework demonstrates a truly scalable and sustainable solution for complex continuous control tasks in real-world applications. 

**Abstract (ZH)**: 在人工智能革命时代，大规模数据驱动的人工智能系统投资需要高性能计算，消耗大量能源和资源。这一趋势提出了在不牺牲可扩展性或性能的情况下优化可持续性的新挑战。作为传统冯·诺依曼架构的节能替代方案，神经形态计算及其脉冲神经网络（SNNs）因其固有的能效而颇具前景。然而，在复杂的连续控制任务等实际应用场景中，SNNs往往缺乏传统人工神经网络的性能优化。研究人员通过将SNNs与深度强化学习（DeepRL）相结合来解决这一问题，但可扩展性尚未被探索。本文扩展了我们关于SpikeRL的先前工作，这是一个基于DeepRL的SNNs连续控制的可扩展和节能框架。在我们最初实现的SpikeRL框架中，我们基于Population-coded Spiking Actor Network（PopSAN）方法的群体编码构建了SNN模型，并使用mpi4py通过消息传递接口（MPI）实现了分布式训练，同时通过混合精度训练优化了模型训练。在我们最新的SpikeRL框架中，我们实现了自己的具有群体编码的DeepRL-SNN组件，并使用NCCL后端的PyTorch Distribute包实现了分布式训练，同时仍然通过混合精度训练优化了模型训练。我们新的SpikeRL实现比最先进的DeepRL-SNN方法快4.26倍，节能2.25倍。我们提出的SpikeRL框架展示了在实际应用中为复杂连续控制任务提供真正可扩展和可持续的解决方案。 

---
# External Large Foundation Model: How to Efficiently Serve Trillions of Parameters for Online Ads Recommendation 

**Title (ZH)**: 外部大型基础模型：如何高效地为在线广告推荐服务万亿参数 

**Authors**: Mingfu Liang, Xi Liu, Rong Jin, Boyang Liu, Qiuling Suo, Qinghai Zhou, Song Zhou, Laming Chen, Hua Zheng, Zhiyuan Li, Shali Jiang, Jiyan Yang, Xiaozhen Xia, Fan Yang, Yasmine Badr, Ellie Wen, Shuyu Xu, Hansey Chen, Zhengyu Zhang, Jade Nie, Chunzhi Yang, Zhichen Zeng, Weilin Zhang, Xingliang Huang, Qianru Li, Shiquan Wang, Evelyn Lyu, Wenjing Lu, Rui Zhang, Wenjun Wang, Jason Rudy, Mengyue Hang, Kai Wang, Yinbin Ma, Shuaiwen Wang, Sihan Zeng, Tongyi Tang, Xiaohan Wei, Longhao Jin, Jamey Zhang, Marcus Chen, Jiayi Zhang, Angie Huang, Chi Zhang, Zhengli Zhao, Jared Yang, Qiang Jin, Xian Chen, Amit Anand Amlesahwaram, Lexi Song, Liang Luo, Yuchen Hao, Nan Xiao, Yavuz Yetim, Luoshang Pan, Gaoxiang Liu, Yuxi Hu, Yuzhen Huang, Jackie Xu, Rich Zhu, Xin Zhang, Yiqun Liu, Hang Yin, Yuxin Chen, Buyun Zhang, Xiaoyi Liu, Sylvia Wang, Wenguang Mao, Zhijing Li, Qin Huang, Chonglin Sun, Shupin Mao, Jingzheng Qin, Peggy Yao, Jae-Woo Choi, Bin Gao, Ernest Wang, Lei Zhang, Wen-Yen Chen, Ted Lee, Jay Zha, Yi Meng, Alex Gong, Edison Gao, Alireza Vahdatpour, Yiping Han, Yantao Yao, Toshinari Kureha, Shuo Chang, Musharaf Sultan, John Bocharov, Sagar Chordia, Xiaorui Gan, Peng Sun, Rocky Liu, Bo Long, Wenlin Chen, Santanu Kolay, Huayu Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.17494)  

**Abstract**: Ads recommendation is a prominent service of online advertising systems and has been actively studied. Recent studies indicate that scaling-up and advanced design of the recommendation model can bring significant performance improvement. However, with a larger model scale, such prior studies have a significantly increasing gap from industry as they often neglect two fundamental challenges in industrial-scale applications. First, training and inference budgets are restricted for the model to be served, exceeding which may incur latency and impair user experience. Second, large-volume data arrive in a streaming mode with data distributions dynamically shifting, as new users/ads join and existing users/ads leave the system. We propose the External Large Foundation Model (ExFM) framework to address the overlooked challenges. Specifically, we develop external distillation and a data augmentation system (DAS) to control the computational cost of training/inference while maintaining high performance. We design the teacher in a way like a foundation model (FM) that can serve multiple students as vertical models (VMs) to amortize its building cost. We propose Auxiliary Head and Student Adapter to mitigate the data distribution gap between FM and VMs caused by the streaming data issue. Comprehensive experiments on internal industrial-scale applications and public datasets demonstrate significant performance gain by ExFM. 

**Abstract (ZH)**: 外部大型基础模型框架（ExFM）：应对工业规模应用中的挑战 

---
# Pursuing Top Growth with Novel Loss Function 

**Title (ZH)**: 追求新型损失函数下的顶级增长 

**Authors**: Ruoyu Guo, Haochen Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17493)  

**Abstract**: Making consistently profitable financial decisions in a continuously evolving and volatile stock market has always been a difficult task. Professionals from different disciplines have developed foundational theories to anticipate price movement and evaluate securities such as the famed Capital Asset Pricing Model (CAPM). In recent years, the role of artificial intelligence (AI) in asset pricing has been growing. Although the black-box nature of deep learning models lacks interpretability, they have continued to solidify their position in the financial industry. We aim to further enhance AI's potential and utility by introducing a return-weighted loss function that will drive top growth while providing the ML models a limited amount of information. Using only publicly accessible stock data (open/close/high/low, trading volume, sector information) and several technical indicators constructed from them, we propose an efficient daily trading system that detects top growth opportunities. Our best models achieve 61.73% annual return on daily rebalancing with an annualized Sharpe Ratio of 1.18 over 1340 testing days from 2019 to 2024, and 37.61% annual return with an annualized Sharpe Ratio of 0.97 over 1360 testing days from 2005 to 2010. The main drivers for success, especially independent of any domain knowledge, are the novel return-weighted loss function, the integration of categorical and continuous data, and the ML model architecture. We also demonstrate the superiority of our novel loss function over traditional loss functions via several performance metrics and statistical evidence. 

**Abstract (ZH)**: 在不断变化和波动的股票市场上做出一致盈利的金融决策一直是一项艰巨的任务。来自不同学科的专业人士开发了基础理论来预测价格波动和评估证券，例如著名的资本资产定价模型（CAPM）。近年来，人工智能（AI）在资产定价中的作用日益增长。尽管深度学习模型的黑箱性质缺乏解释性，它们在金融行业中的地位仍然得到了巩固。我们通过引入一种基于收益加权的损失函数来进一步增强AI的潜力和实用性，该函数将驱动顶级增长同时为ML模型提供有限的信息。仅使用公开可获取的股票数据（开盘价/收盘价/最高价/最低价、交易量、行业信息）以及从中构建的若干技术指标，我们提出了一种高效的每日交易系统，以检测顶级增长机会。我们的最佳模型在2019年至2024年的1340个测试日实现了年化61.73%的回报率和年化Sharpe比率1.18，在2005年至2010年的1360个测试日实现了年化37.61%的回报率和年化Sharpe比率0.97。尤其是在没有领域知识的情况下，成功的主要驱动因素是新颖的收益加权损失函数、类别数据和连续数据的整合以及ML模型架构。我们还通过多种性能指标和统计证据展示了我们新型损失函数相对于传统损失函数的优势。 

---
# A generalized dual potential for inelastic Constitutive Artificial Neural Networks: A JAX implementation at finite strains 

**Title (ZH)**: 广义对偶势函数在有限应变下对非弹性本构人工神经网络的应用：基于JAX的实现 

**Authors**: Hagen Holthusen, Kevin Linka, Ellen Kuhl, Tim Brepols  

**Link**: [PDF](https://arxiv.org/pdf/2502.17490)  

**Abstract**: We present a methodology for designing a generalized dual potential, or pseudo potential, for inelastic Constitutive Artificial Neural Networks (iCANNs). This potential, expressed in terms of stress invariants, inherently satisfies thermodynamic consistency for large deformations. In comparison to our previous work, the new potential captures a broader spectrum of material behaviors, including pressure-sensitive inelasticity.
To this end, we revisit the underlying thermodynamic framework of iCANNs for finite strain inelasticity and derive conditions for constructing a convex, zero-valued, and non-negative dual potential. To embed these principles in a neural network, we detail the architecture's design, ensuring a priori compliance with thermodynamics.
To evaluate the proposed architecture, we study its performance and limitations discovering visco-elastic material behavior, though the method is not limited to visco-elasticity. In this context, we investigate different aspects in the strategy of discovering inelastic materials. Our results indicate that the novel architecture robustly discovers interpretable models and parameters, while autonomously revealing the degree of inelasticity.
The iCANN framework, implemented in JAX, is publicly accessible at this https URL. 

**Abstract (ZH)**: 一种广义双重势能的设计方法：应用于不可积Artificial Neural Networks的非弹性行为 

---
# Using Graph Convolutional Networks to Address fMRI Small Data Problems 

**Title (ZH)**: 使用图卷积网络解决fMRI小数据问题 

**Authors**: Thomas Screven, Andras Necz, Jason Smucny, Ian Davidson  

**Link**: [PDF](https://arxiv.org/pdf/2502.17489)  

**Abstract**: Although great advances in the analysis of neuroimaging data have been made, a major challenge is a lack of training data. This is less problematic in tasks such as diagnosis, where much data exists, but particularly prevalent in harder problems such as predicting treatment responses (prognosis), where data is focused and hence limited. Here, we address the learning from small data problems for medical imaging using graph neural networks. This is particularly challenging as the information about the patients is themselves graphs (regions of interest connectivity graphs). We show how a spectral representation of the connectivity data allows for efficient propagation that can yield approximately 12\% improvement over traditional deep learning methods using the exact same data. We show that our method's superior performance is due to a data smoothing result that can be measured by closing the number of triangle inequalities and thereby satisfying transitivity. 

**Abstract (ZH)**: 尽管在神经影像数据分析方面取得了巨大进展，但训练数据不足仍然是一个主要挑战。在诊断这类任务中，由于数据丰富，这一问题较不突出，但在预测治疗反应（预后）这类更难的问题中，数据集中且有限，问题尤为突出。为了解决小数据学习问题，我们使用图神经网络处理医学影像。由于患者信息本身即为图（感兴趣区域连接图），这一问题尤为具有挑战性。我们展示了如何通过谱表示连接数据来实现高效传播，这可以使我们的方法在使用相同数据的情况下，比传统深度学习方法性能提高约12%。我们证明，我们的方法性能优越的原因在于一种数据平滑效果，这可以通过减少三角不等式的数量来衡量，并因此满足传递性。 

---
# Toward Foundational Model for Sleep Analysis Using a Multimodal Hybrid Self-Supervised Learning Framework 

**Title (ZH)**: 面向睡眠分析的多模态混合自监督学习框架的基础模型研究 

**Authors**: Cheol-Hui Lee, Hakseung Kim, Byung C. Yoon, Dong-Joo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.17481)  

**Abstract**: Sleep is essential for maintaining human health and quality of life. Analyzing physiological signals during sleep is critical in assessing sleep quality and diagnosing sleep disorders. However, manual diagnoses by clinicians are time-intensive and subjective. Despite advances in deep learning that have enhanced automation, these approaches remain heavily dependent on large-scale labeled datasets. This study introduces SynthSleepNet, a multimodal hybrid self-supervised learning framework designed for analyzing polysomnography (PSG) data. SynthSleepNet effectively integrates masked prediction and contrastive learning to leverage complementary features across multiple modalities, including electroencephalogram (EEG), electrooculography (EOG), electromyography (EMG), and electrocardiogram (ECG). This approach enables the model to learn highly expressive representations of PSG data. Furthermore, a temporal context module based on Mamba was developed to efficiently capture contextual information across signals. SynthSleepNet achieved superior performance compared to state-of-the-art methods across three downstream tasks: sleep-stage classification, apnea detection, and hypopnea detection, with accuracies of 89.89%, 99.75%, and 89.60%, respectively. The model demonstrated robust performance in a semi-supervised learning environment with limited labels, achieving accuracies of 87.98%, 99.37%, and 77.52% in the same tasks. These results underscore the potential of the model as a foundational tool for the comprehensive analysis of PSG data. SynthSleepNet demonstrates comprehensively superior performance across multiple downstream tasks compared to other methodologies, making it expected to set a new standard for sleep disorder monitoring and diagnostic systems. 

**Abstract (ZH)**: 睡眠对于维持人类健康和生活质量至关重要。分析睡眠期间的生理信号是评估睡眠质量和诊断睡眠障碍的关键。然而，临床医生的手动诊断耗时且主观。尽管深度学习的进步提高了自动化程度，但这些方法仍高度依赖大规模标注数据集。本研究引入了SynthSleepNet，这是一种多模态混合自监督学习框架，用于分析多导生理记录图（PSG）数据。SynthSleepNet有效结合了掩膜预测和对比学习，充分利用多种模态的互补特征，包括脑电图（EEG）、眼电图（EOG）、肌电图（EMG）和心电图（ECG）。该方法使模型能够学习 PSG 数据的高表现力表示。此外，基于 Mamba 开发了一种时间上下文模块，以高效捕捉信号间的上下文信息。SynthSleepNet 在三项下游任务（睡眠阶段分类、呼吸暂停检测和低通气检测）中分别实现了 89.89%、99.75% 和 89.60% 的准确率，表现出色。在有限标注数据的半监督学习环境中，模型同样表现出色，在同一任务中分别实现了 87.98%、99.37% 和 77.52% 的准确率。这些结果突显了该模型作为 PSG 数据全面分析基础工具的潜力。SynthSleepNet 在多个下游任务中的综合性能优于其他方法，预计将为睡眠障碍监测和诊断系统设定新的标准。 

---
# Brain-to-Text Decoding: A Non-invasive Approach via Typing 

**Title (ZH)**: 脑到文本解码：一种通过打字的无侵入性方法 

**Authors**: Jarod Lévy, Mingfang Zhang, Svetlana Pinet, Jérémy Rapin, Hubert Banville, Stéphane d'Ascoli, Jean-Rémi King  

**Link**: [PDF](https://arxiv.org/pdf/2502.17480)  

**Abstract**: Modern neuroprostheses can now restore communication in patients who have lost the ability to speak or move. However, these invasive devices entail risks inherent to neurosurgery. Here, we introduce a non-invasive method to decode the production of sentences from brain activity and demonstrate its efficacy in a cohort of 35 healthy volunteers. For this, we present Brain2Qwerty, a new deep learning architecture trained to decode sentences from either electro- (EEG) or magneto-encephalography (MEG), while participants typed briefly memorized sentences on a QWERTY keyboard. With MEG, Brain2Qwerty reaches, on average, a character-error-rate (CER) of 32% and substantially outperforms EEG (CER: 67%). For the best participants, the model achieves a CER of 19%, and can perfectly decode a variety of sentences outside of the training set. While error analyses suggest that decoding depends on motor processes, the analysis of typographical errors suggests that it also involves higher-level cognitive factors. Overall, these results narrow the gap between invasive and non-invasive methods and thus open the path for developing safe brain-computer interfaces for non-communicating patients. 

**Abstract (ZH)**: 非侵入性方法从脑活动解码句子以恢复沟通：一项在35名健康志愿者中的有效性演示 

---
# ECG-Expert-QA: A Benchmark for Evaluating Medical Large Language Models in Heart Disease Diagnosis 

**Title (ZH)**: ECG-Expert-QA：用于心脏疾病诊断的大规模语言模型评估基准 

**Authors**: Xu Wang, Jiaju Kang, Puyu Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.17475)  

**Abstract**: We present ECG-Expert-QA, a comprehensive multimodal dataset designed for evaluating diagnostic capabilities in ECG interpretation, integrating real clinical data with systematically generated synthetic cases. The dataset encompasses six fundamental diagnostic tasks, comprising 47,211 meticulously curated question-answer pairs that span a spectrum of clinical scenarios, from basic rhythm analysis to complex case interpretation. By simulating challenging clinical cases through a rigorous medical knowledge-guided process, ECG-Expert-QA not only enhances the availability of annotated diagnostic data but also significantly increases the complexity and diversity of clinical presentations, including rare cardiac conditions and temporal progression patterns. This design enables comprehensive evaluation of medical language models across multiple dimensions, including diagnostic accuracy, clinical reasoning, and knowledge integration. To facilitate global research collaboration, ECG-Expert-QA is available in both Chinese and English versions, with rigorous quality control ensuring linguistic and clinical consistency. The dataset's challenging diagnostic tasks, which include interpretation of complex arrhythmias, identification of subtle ischemic changes, and integration of clinical context, establish it as an effective benchmark for advancing AI-assisted ECG interpretation and pushing the boundaries of current diagnostic models. Our dataset is open-source and available at this https URL. 

**Abstract (ZH)**: ECG-Expert-QA：一个全面的多模态数据集，用于评估ECG解释的诊断能力，结合实际临床数据与系统生成的合成病例。 

---
# MC2SleepNet: Multi-modal Cross-masking with Contrastive Learning for Sleep Stage Classification 

**Title (ZH)**: MC2SleepNet：多模态跨掩蔽对比学习的睡眠阶段分类 

**Authors**: Younghoon Na  

**Link**: [PDF](https://arxiv.org/pdf/2502.17470)  

**Abstract**: Sleep profoundly affects our health, and sleep deficiency or disorders can cause physical and mental problems. % Despite significant findings from previous studies, challenges persist in optimizing deep learning models, especially in multi-modal learning for high-accuracy sleep stage classification. Our research introduces MC2SleepNet (Multi-modal Cross-masking with Contrastive learning for Sleep stage classification Network). It aims to facilitate the effective collaboration between Convolutional Neural Networks (CNNs) and Transformer architectures for multi-modal training with the help of contrastive learning and cross-masking. % Raw single channel EEG signals and corresponding spectrogram data provide differently characterized modalities for multi-modal learning. Our MC2SleepNet has achieved state-of-the-art performance with an accuracy of both 84.6% on the SleepEDF-78 and 88.6% accuracy on the Sleep Heart Health Study (SHHS). These results demonstrate the effective generalization of our proposed network across both small and large datasets. 

**Abstract (ZH)**: 睡眠深刻影响我们的健康，睡眠不足或障碍会导致身体和心理问题。尽管先前研究取得了显著成果，但在优化深度学习模型，尤其是高精度睡眠阶段分类的多模态学习中仍存在挑战。我们的研究引入了MC2SleepNet（多模态交叉遮蔽与对比学习网络），旨在通过对比学习和交叉遮蔽促进卷积神经网络（CNNs）与变压器架构之间的有效协作，实现多模态训练。原始单通道EEG信号及其相应的 spectrogram 数据为多模态学习提供了不同特征的模态。我们的MC2SleepNet在SleepEDF-78数据集上达到了84.6%的准确率，在睡眠心脏健康研究（SHHS）数据集上达到了88.6%的准确率。这些结果表明，我们提出的网络在小规模和大规模数据集上的有效泛化能力。 

---
# PixleepFlow: A Pixel-Based Lifelog Framework for Predicting Sleep Quality and Stress Level 

**Title (ZH)**: 基于像素的 lifelog 框架：预测睡眠质量与压力水平 

**Authors**: Younghoon Na  

**Link**: [PDF](https://arxiv.org/pdf/2502.17469)  

**Abstract**: The analysis of lifelogs can yield valuable insights into an individual's daily life, particularly with regard to their health and well-being. The accurate assessment of quality of life is necessitated by the use of diverse sensors and precise synchronization. To rectify this issue, this study proposes the image-based sleep quality and stress level estimation flow (PixleepFlow). PixleepFlow employs a conversion methodology into composite image data to examine sleep patterns and their impact on overall health. Experiments were conducted using lifelog datasets to ascertain the optimal combination of data formats. In addition, we identified which sensor information has the greatest influence on the quality of life through Explainable Artificial Intelligence(XAI). As a result, PixleepFlow produced more significant results than various data formats. This study was part of a written-based competition, and the additional findings from the lifelog dataset are detailed in Section Section IV. More information about PixleepFlow can be found at this https URL. 

**Abstract (ZH)**: 基于图像的睡眠质量和压力水平估计流程（PixleepFlow）对日常生活的影响分析 

---
# The Case for Cleaner Biosignals: High-fidelity Neural Compressor Enables Transfer from Cleaner iEEG to Noisier EEG 

**Title (ZH)**: cleaner 生物信号的重要性：高保真神经压缩器使 cleaner iEEG 能够向 noisier EEG 转移 

**Authors**: Francesco Stefano Carzaniga, Gary Tom Hoppeler, Michael Hersche, Kaspar Anton Schindler, Abbas Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2502.17462)  

**Abstract**: All data modalities are not created equal, even when the signal they measure comes from the same source. In the case of the brain, two of the most important data modalities are the scalp electroencephalogram (EEG), and the intracranial electroencephalogram (iEEG). They are used by human experts, supported by deep learning (DL) models, to accomplish a variety of tasks, such as seizure detection and motor imagery classification. Although the differences between EEG and iEEG are well understood by human experts, the performance of DL models across these two modalities remains under-explored. To help characterize the importance of clean data on the performance of DL models, we propose BrainCodec, a high-fidelity EEG and iEEG neural compressor. We find that training BrainCodec on iEEG and then transferring to EEG yields higher reconstruction quality than training on EEG directly. In addition, we also find that training BrainCodec on both EEG and iEEG improves fidelity when reconstructing EEG. Our work indicates that data sources with higher SNR, such as iEEG, provide better performance across the board also in the medical time-series domain. BrainCodec also achieves up to a 64x compression on iEEG and EEG without a notable decrease in quality. BrainCodec markedly surpasses current state-of-the-art compression models both in final compression ratio and in reconstruction fidelity. We also evaluate the fidelity of the compressed signals objectively on a seizure detection and a motor imagery task performed by standard DL models. Here, we find that BrainCodec achieves a reconstruction fidelity high enough to ensure no performance degradation on the downstream tasks. Finally, we collect the subjective assessment of an expert neurologist, that confirms the high reconstruction quality of BrainCodec in a realistic scenario. The code is available at this https URL. 

**Abstract (ZH)**: 不同类型的脑电数据并非同等重要，即使它们源自相同的源。在大脑成像中，头皮脑电图（EEG）和颅内脑电图（iEEG）是最为重要的两种数据模态。人类专家和深度学习（DL）模型利用这些数据模态完成各种任务，例如癫痫检测和运动想象分类。尽管人类专家对EEG和iEEG之间的差异有深入的理解，但这些两种模态下DL模型的性能尚待探索。为了帮助确定干净数据对DL模型性能的重要性，我们提出了脑编码器（BrainCodec），一种高保真EEG和iEEG神经压缩器。我们发现，使用iEEG训练BrainCodec然后转换到EEG进行训练，可以得到比直接使用EEG训练更高的重建质量。此外，我们还发现，使用EEG和iEEG两种数据模态训练BrainCodec，可以提高重建EEG的保真度。我们的研究结果显示，信噪比更高的数据源，如iEEG，在医学时间序列领域也能提供更优的整体性能。BrainCodec在iEEG和EEG上的压缩比高达64倍，同时保真度无明显下降。BrainCodec在最终压缩比和重建保真度上均显著优于当前最先进的压缩模型。我们还客观地评估了标准DL模型执行的癫痫检测和运动想象任务中压缩信号的保真度。结果显示，BrainCodec的重建保真度足够高，以确保下游任务的性能不受影响。最后，我们收集了一名专家神经学家的主观评估，证实了BrainCodec在实际场景中的高重建质量。代码可在以下链接获取。 

---
# Finetuning and Quantization of EEG-Based Foundational BioSignal Models on ECG and PPG Data for Blood Pressure Estimation 

**Title (ZH)**: 基于EEG的生物信号基础模型在ECG和PPG数据上的微调与量化以估计血压 

**Authors**: Bálint Tóth, Dominik Senti, Thorir Mar Ingolfsson, Jeffrey Zweidler, Alexandre Elsig, Luca Benini, Yawei Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.17460)  

**Abstract**: Blood pressure (BP) is a key indicator of cardiovascular health. As hypertension remains a global cause of morbidity and mortality, accurate, continuous, and non-invasive BP monitoring is therefore of paramount importance. Photoplethysmography (PPG) and electrocardiography (ECG) can potentially enable continuous BP monitoring, yet training accurate and robust machine learning (ML) models remains challenging due to variability in data quality and patient-specific factors. Recently, multiple research groups explored Electroencephalographic (EEG)--based foundation models and demonstrated their exceptional ability to learn rich temporal resolution. Considering the morphological similarities between different biosignals, the question arises of whether a model pre-trained on one modality can effectively be exploited to improve the accuracy of a different signal type. In this work, we take an initial step towards generalized biosignal foundation models by investigating whether model representations learned from abundant EEG data can effectively be transferred to ECG/PPG data solely with fine-tuning, without the need for large-scale additional pre-training, for the BP estimation task. Evaluations on the MIMIC-III and VitalDB datasets demonstrate that our approach achieves near state-of-the-art accuracy for diastolic BP (mean absolute error of 1.57 mmHg) and surpasses by 1.5x the accuracy of prior works for systolic BP (mean absolute error 2.72 mmHg). Additionally, we perform dynamic INT8 quantization, reducing the smallest model size by over 3.5x (from 13.73 MB down to 3.83 MB) while preserving performance, thereby enabling unobtrusive, real-time BP monitoring on resource-constrained wearable devices. 

**Abstract (ZH)**: 基于脑电图的通用生物信号基础模型：EditTextlightlyadaptedfromabundanteegdatatofine-tuneecg/ppgdatasforbloodpressureestimation 

---
# MoEMba: A Mamba-based Mixture of Experts for High-Density EMG-based Hand Gesture Recognition 

**Title (ZH)**: MoEMba：一种基于Mamba的专家混合模型高密度EMG_hand手势识别 

**Authors**: Mehran Shabanpour, Kasra Rad, Sadaf Khademi, Arash Mohammadi  

**Link**: [PDF](https://arxiv.org/pdf/2502.17457)  

**Abstract**: High-Density surface Electromyography (HDsEMG) has emerged as a pivotal resource for Human-Computer Interaction (HCI), offering direct insights into muscle activities and motion intentions. However, a significant challenge in practical implementations of HD-sEMG-based models is the low accuracy of inter-session and inter-subject classification. Variability between sessions can reach up to 40% due to the inherent temporal variability of HD-sEMG signals. Targeting this challenge, the paper introduces the MoEMba framework, a novel approach leveraging Selective StateSpace Models (SSMs) to enhance HD-sEMG-based gesture recognition. The MoEMba framework captures temporal dependencies and cross-channel interactions through channel attention techniques. Furthermore, wavelet feature modulation is integrated to capture multi-scale temporal and spatial relations, improving signal representation. Experimental results on the CapgMyo HD-sEMG dataset demonstrate that MoEMba achieves a balanced accuracy of 56.9%, outperforming its state-of-the-art counterparts. The proposed framework's robustness to session-to-session variability and its efficient handling of high-dimensional multivariate time series data highlight its potential for advancing HD-sEMG-powered HCI systems. 

**Abstract (ZH)**: 高密度表面肌电图（HDsEMG）在人机交互（HCI）中的新兴资源及其MoEMba框架：利用选择性状态空间模型提升手势识别 

---
# Survey on Recent Progress of AI for Chemistry: Methods, Applications, and Opportunities 

**Title (ZH)**: 近期AI在化学领域进展调研：方法、应用及机遇 

**Authors**: Ding Hu, Pengxiang Hua, Zhen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17456)  

**Abstract**: The development of artificial intelligence (AI) techniques has brought revolutionary changes across various realms. In particular, the use of AI-assisted methods to accelerate chemical research has become a popular and rapidly growing trend, leading to numerous groundbreaking works. In this paper, we provide a comprehensive review of current AI techniques in chemistry from a computational perspective, considering various aspects in the design of methods. We begin by discussing the characteristics of data from diverse sources, followed by an overview of various representation methods. Next, we review existing models for several topical tasks in the field, and conclude by highlighting some key challenges that warrant further attention. 

**Abstract (ZH)**: 人工智能技术的发展在各个领域带来了革命性的变化，特别是AI辅助方法加速化学研究已成为一个流行且迅速增长的趋势，产生了众多开创性的工作。本文从计算视角对当前化学中的AI技术进行了全面回顾，考虑了方法设计的各个方面。首先讨论来自各种来源的数据特性，随后概述各种表示方法，接着回顾了该领域几个专题任务的现有模型，并最后强调了一些值得进一步关注的关键挑战。 

---
# Smart Sampling Strategies for Wireless Industrial Data Acquisition 

**Title (ZH)**: 智能无线工业数据采集的智能采样策略 

**Authors**: Marcos Soto  

**Link**: [PDF](https://arxiv.org/pdf/2502.17454)  

**Abstract**: In industrial environments, data acquisition accuracy is crucial for process control and optimization. Wireless telemetry has proven to be a valuable tool for improving efficiency in well-testing operations, enabling bidirectional communication and real-time control of downhole tools. However, high sampling frequencies present challenges in telemetry, including data storage, transmission, computational resource consumption, and battery life of wireless devices. This study explores how optimizing data acquisition strategies can reduce aliasing effects and systematic errors while improving sampling rates without compromising measurement accuracy. A reduction of 80% in sampling frequency was achieved without degrading measurement quality, demonstrating the potential for resource optimization in industrial environments. 

**Abstract (ZH)**: 在工业环境中的数据采集精度对于过程控制和优化至关重要。无线遥测已被证明是提高油井测试操作效率的有效工具，能够实现双向通信和井下工具的实时控制。然而，高采样频率给遥测带来了挑战，包括数据存储、传输、计算资源消耗以及无线设备的电池寿命。本研究探讨了如何通过优化数据采集策略来减少抽样频率、降低aliasing效应和系统误差，同时提高采样率而不牺牲测量精度。结果表明，可以在不降低测量质量的情况下将采样频率减少80%，展示了在工业环境中资源优化的潜力。 

---
# AirTag, You're It: Reverse Logistics and Last Mile Dynamics 

**Title (ZH)**: AirTag,你负责：逆向物流与最后一公里动态 

**Authors**: David Noever, Forrest McKee  

**Link**: [PDF](https://arxiv.org/pdf/2502.17447)  

**Abstract**: This study addresses challenges in reverse logistics, a frequently overlooked but essential component of last-mile delivery, particularly in disaster relief scenarios where infrastructure disruptions demand adaptive solutions. While hub-and-spoke logistics networks excel at long-distance scalability, they often fail to optimize closely spaced spokes reliant on distant hubs, introducing inefficiencies in transit times and resource allocation. Using 20 Apple AirTags embedded in packages, this research provides empirical insights into logistical flows, capturing granular spatial and temporal data through Bluetooth LE (BLE) 5 trackers integrated with the Apple Find My network. These trackers demonstrated their value in monitoring dynamic cargo movements, enabling real-time adjustments in mobile hub placement and route optimization, particularly in disaster relief contexts like Hurricane Helene. A novel application of discrete event simulation (DES) further explored the saddle point in hub-spoke configurations, where excessive hub reliance clashes with diminishing spoke interaction demand. By coupling simulation results with empirical AirTag tracking, the study highlights the potential of BLE technology to refine reverse logistics, reduce delays, and improve operational flexibility in both routine and crisis-driven delivery networks. 

**Abstract (ZH)**: 本研究探讨了逆向物流面临的挑战，这是getLastmile配送中一个经常被忽视但至关重要的组成部分，特别是在基础设施中断需求适应性解决方案的灾后救援场景中。尽管枢纽和辐条物流网络在长距离方面表现出色，但在优化紧密布设而依赖远端枢纽的辐条时，它们往往引入了传输时间和资源配置的低效率。利用20个集成在包裹中的Apple AirTags，本研究提供了实证见解，通过BLE 5追踪器集成Apple Find My网络捕获了详细的时空数据。这些追踪器展示了其在监测动态货物移动中的价值，允许在移动枢纽放置和路线优化方面进行实时调整，特别是在飓风海伦这样的灾后救援情境中。通过将离散事件模拟（DES）与新型应用相结合，进一步探讨了枢纽和辐条配置中的鞍点，即过度依赖枢纽与辐条间互动需求下降之间的冲突。通过将模拟结果与AirTag追踪数据结合，本研究突显了BLE技术在改进逆向物流、减少延误以及提高常规和危机驱动配送网络操作灵活性方面的潜力。 

---
# DCentNet: Decentralized Multistage Biomedical Signal Classification using Early Exits 

**Title (ZH)**: DCentNet: 分布式多阶段生物医学信号分类方法及早退出 

**Authors**: Xiaolin Li, Binhua Huang, Barry Cardiff, Deepu John  

**Link**: [PDF](https://arxiv.org/pdf/2502.17446)  

**Abstract**: DCentNet is a novel decentralized multistage signal classification approach designed for biomedical data from IoT wearable sensors, integrating early exit points (EEP) to enhance energy efficiency and processing speed. Unlike traditional centralized processing methods, which result in high energy consumption and latency, DCentNet partitions a single CNN model into multiple sub-networks using EEPs. By introducing encoder-decoder pairs at EEPs, the system compresses large feature maps before transmission, significantly reducing wireless data transfer and power usage. If an input is confidently classified at an EEP, processing stops early, optimizing efficiency. Initial sub-networks can be deployed on fog or edge devices to further minimize energy consumption. A genetic algorithm is used to optimize EEP placement, balancing performance and complexity. Experimental results on ECG classification show that with one EEP, DCentNet reduces wireless data transmission by 94.54% and complexity by 21%, while maintaining original accuracy and sensitivity. With two EEPs, sensitivity reaches 98.36%, accuracy 97.74%, wireless data transmission decreases by 91.86%, and complexity is reduced by 22%. Implemented on an ARM Cortex-M4 MCU, DCentNet achieves an average power saving of 73.6% compared to continuous wireless ECG transmission. 

**Abstract (ZH)**: DCentNet：一种集成早期退出点的分散多阶段生物医学信号分类方法 

---
# Interpretable Dual-Filter Fuzzy Neural Networks for Affective Brain-Computer Interfaces 

**Title (ZH)**: 可解释的双滤波模糊神经网络在情绪脑机接口中的应用 

**Authors**: Xiaowei Jiang, Yanan Chen, Nikhil Ranjan Pal, Yu-Cheng Chang, Yunkai Yang, Thomas Do, Chin-Teng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.17445)  

**Abstract**: Fuzzy logic provides a robust framework for enhancing explainability, particularly in domains requiring the interpretation of complex and ambiguous signals, such as brain-computer interface (BCI) systems. Despite significant advances in deep learning, interpreting human emotions remains a formidable challenge. In this work, we present iFuzzyAffectDuo, a novel computational model that integrates a dual-filter fuzzy neural network architecture for improved detection and interpretation of emotional states from neuroimaging data. The model introduces a new membership function (MF) based on the Laplace distribution, achieving superior accuracy and interpretability compared to traditional approaches. By refining the extraction of neural signals associated with specific emotions, iFuzzyAffectDuo offers a human-understandable framework that unravels the underlying decision-making processes. We validate our approach across three neuroimaging datasets using functional Near-Infrared Spectroscopy (fNIRS) and Electroencephalography (EEG), demonstrating its potential to advance affective computing. These findings open new pathways for understanding the neural basis of emotions and their application in enhancing human-computer interaction. 

**Abstract (ZH)**: 模糊逻辑提供了一种增强可解释性的坚实框架，特别是在需要解释复杂和模糊信号的领域，如脑机接口（BCI）系统。尽管深度学习取得了显著进展，但解释人类情感仍然是一个艰巨的挑战。在此项工作中，我们提出了iFuzzyAffectDuo，一种新颖的计算模型，该模型结合了双滤波模糊神经网络架构，以提高情绪状态从神经影像数据中检测和解释的准确性与可解释性。该模型引入了基于拉普拉斯分布的新隶属函数（MF），其准确性和可解释性优于传统方法。通过细化与特定情绪相关的神经信号提取，iFuzzyAffectDuo 提供了一种人类可理解的框架，揭示了潜在的决策过程。我们使用功能性近红外光谱成像（fNIRS）和脑电图（EEG）的三个神经影像数据集验证了该方法，展示了其在促进情感计算方面的潜力。这些发现为理解情绪的神经基础及其在增强人机交互中的应用开辟了新的途径。 

---
# AI Agentic workflows and Enterprise APIs: Adapting API architectures for the age of AI agents 

**Title (ZH)**: AI代理工作流与企业API：适应AI代理时代的数据架构适配 

**Authors**: Vaibhav Tupe, Shrinath Thube  

**Link**: [PDF](https://arxiv.org/pdf/2502.17443)  

**Abstract**: The rapid advancement of Generative AI has catalyzed the emergence of autonomous AI agents, presenting unprecedented challenges for enterprise computing infrastructures. Current enterprise API architectures are predominantly designed for human-driven, predefined interaction patterns, rendering them ill-equipped to support intelligent agents' dynamic, goal-oriented behaviors. This research systematically examines the architectural adaptations for enterprise APIs to support AI agentic workflows effectively. Through a comprehensive analysis of existing API design paradigms, agent interaction models, and emerging technological constraints, the paper develops a strategic framework for API transformation. The study employs a mixed-method approach, combining theoretical modeling, comparative analysis, and exploratory design principles to address critical challenges in standardization, performance, and intelligent interaction. The proposed research contributes a conceptual model for next-generation enterprise APIs that can seamlessly integrate with autonomous AI agent ecosystems, offering significant implications for future enterprise computing architectures. 

**Abstract (ZH)**: 生成式人工智能的快速发展催生了自主人工智能代理的出现，为企业计算基础设施带来了前所未有的挑战。当前的企业API架构主要针对由人类驱动的预定义交互模式设计，使其无法有效支持智能代理的动态、目标导向的行为。本文系统地探讨了企业API架构的适配，以有效支持人工智能代理工作流。通过全面分析现有的API设计范式、代理交互模型和新兴的技术约束，该论文提出了API转型的战略框架。研究采用了混合方法，结合理论建模、比较分析和探索性设计原则，以应对标准制定、性能和智能交互的关键挑战。提出的研究所提供的概念模型可以无缝集成到自主人工智能代理生态系统中，并对未来的企业计算架构具有重要意义。 

---
# Thinking Before Running! Efficient Code Generation with Thorough Exploration and Optimal Refinement 

**Title (ZH)**: 深思而后行！通过全面探索与最优细化实现高效代码生成 

**Authors**: Xiaoqing Zhang, Yuhan Liu, Flood Sung, Xiuying Chen, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2502.17442)  

**Abstract**: Code generation is crucial in software engineering for automating the coding process efficiently. While test-time computation methods show promise, they suffer from high latency due to multiple computation rounds. To overcome this, we introduce ThinkCoder, a framework that combines thorough exploration with optimal refinement. The exploration phase diversifies the solution space by searching for potential solutions, followed by a refinement phase that enhances precision. This approach allows us to select the best solution through careful consideration before taking action, avoiding excessive trial and error. To further minimize test-time computation overhead, we introduce preference-driven optimization with Reinforced Self-Training (ReST), which uses exploration trajectories from ThinkCoder to guide LLM's evolution. By learning preferences, this approach improves LLM's exploration efficiency, reducing computational costs while maintaining accuracy. ThinkCoder boosts the performance of multiple base LLMs, excelling on benchmarks like HumanEval and MBPP. Compared to SOTA models, it improves Pass@1 by 1.5\% over MapCoder with just 21.7\% of the computation cost. Against AgentCoder, ThinkCoder achieves a 0.6\% higher Pass@1 after 2 rounds, outperforming AgentCoder's 5 rounds. Additionally, ReST with success trajectories enhances efficiency, allowing models like LLaMA2-7B to achieve competitive results using only 20\% of the computational resources. These results highlight the framework's effectiveness and scalability. 

**Abstract (ZH)**: 代码生成对于软件工程中的自动化编码过程至关重要。尽管测试时的计算方法显示出潜力，但由于需要多轮计算，它们会遭受高延迟的问题。为解决这一问题，我们引入了ThinkCoder框架，该框架结合了深入的探索和最优的精炼。探索阶段通过寻找潜在解决方案来多样化解空间，随后的精炼阶段则提高精确度。这种方法使我们能够在采取行动之前仔细考虑并选择最佳解决方案，避免过多的试错。为了进一步减少测试时的计算开销，我们引入了基于偏好的优化与强化自我训练（ReST）方法，该方法利用ThinkCoder的探索轨迹来引导LLM的进化。通过学习偏好，这种方法提高了LLM的探索效率，降低了计算成本同时保持准确性。ThinkCoder提高了多个基础LLM的性能，在HumanEval和MBPP等基准测试中表现出色。与SOTA模型相比，ThinkCoder使用仅21.7%的计算成本就能将MapCoder的Pass@1提高1.5%。与AgentCoder相比，ThinkCoder在2轮后将Pass@1提高了0.6%，优于AgentCoder的5轮。此外，使用成功轨迹的ReST使得模型如LLaMA2-7B仅使用20%的计算资源就能达到竞争力的结果。这些结果突显了该框架的有效性和可扩展性。 

---
# Large Language Models as Realistic Microservice Trace Generators 

**Title (ZH)**: 大规模语言模型作为现实微服务跟踪生成器 

**Authors**: Donghyun Kim, Sriram Ravula, Taemin Ha, Alexandros G. Dimakis, Daehyeok Kim, Aditya Akella  

**Link**: [PDF](https://arxiv.org/pdf/2502.17439)  

**Abstract**: Computer system workload traces, which record hardware or software events during application execution, are essential for understanding the behavior of complex systems and managing their processing and memory resources. However, obtaining real-world traces can be challenging due to the significant collection overheads in performance and privacy concerns that arise in proprietary systems. As a result, synthetic trace generation is considered a promising alternative to using traces collected in real-world production deployments. This paper proposes to train a large language model (LLM) to generate synthetic workload traces, specifically microservice call graphs. To capture complex and arbitrary hierarchical structures and implicit constraints in such traces, we fine-tune LLMs to generate each layer recursively, making call graph generation a sequence of easier steps. To further enforce learning constraints in traces and generate uncommon situations, we apply additional instruction tuning steps to align our model with the desired trace features. Our evaluation results show that our model can generate diverse realistic traces under various conditions and outperform existing methods in accuracy and validity. We show that our synthetically generated traces can effectively substitute real-world data in optimizing or tuning systems management tasks. We also show that our model can be adapted to perform key downstream trace-related tasks, specifically, predicting key trace features and infilling missing data given partial traces. Codes are available in this https URL. 

**Abstract (ZH)**: 计算机系统工作负载轨迹是记录应用执行期间硬件或软件事件的数据，对于理解复杂系统的行为和管理其处理和内存资源至关重要。然而，由于在性能和隐私方面存在显著的收集开销，获取真实世界的轨迹可能具有挑战性。因此，合成轨迹生成被视为使用真实世界生产部署中收集的轨迹的有前景的替代方案。本文提出使用大型语言模型（LLM）生成合成工作负载轨迹，特别是生成微服务调用图。为捕捉此类轨迹中复杂且任意的层次结构和隐含约束，我们递归调优LLM以生成每一层，从而使调用图生成成为一系列较简单步骤。为了进一步强化轨迹中的学习约束并生成不常见的情况，我们应用额外的指令调优步骤，使模型与所需的轨迹特征对齐。我们的评估结果表明，我们的模型可以在各种条件下生成多样化的现实轨迹，并在准确性和有效性方面优于现有方法。我们展示了我们生成的合成轨迹可以有效替代真实世界的数据，以优化或调整系统管理任务。我们还展示了我们的模型可以适应执行关键下游轨迹相关任务，具体而言，预测关键轨迹特征并在部分轨迹下填充缺失数据。代码可在以下链接获得：[这个链接]。 

---
