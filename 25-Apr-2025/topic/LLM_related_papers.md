# Robo-Troj: Attacking LLM-based Task Planners 

**Title (ZH)**: Robo-Troj: 攻击基于LLM的任务规划器 

**Authors**: Mohaiminul Al Nahian, Zainab Altaweel, David Reitano, Sabbir Ahmed, Saumitra Lohokare, Shiqi Zhang, Adnan Siraj Rakin  

**Link**: [PDF](https://arxiv.org/pdf/2504.17070)  

**Abstract**: Robots need task planning methods to achieve goals that require more than individual actions. Recently, large language models (LLMs) have demonstrated impressive performance in task planning. LLMs can generate a step-by-step solution using a description of actions and the goal. Despite the successes in LLM-based task planning, there is limited research studying the security aspects of those systems. In this paper, we develop Robo-Troj, the first multi-trigger backdoor attack for LLM-based task planners, which is the main contribution of this work. As a multi-trigger attack, Robo-Troj is trained to accommodate the diversity of robot application domains. For instance, one can use unique trigger words, e.g., "herical", to activate a specific malicious behavior, e.g., cutting hand on a kitchen robot. In addition, we develop an optimization method for selecting the trigger words that are most effective. Through demonstrating the vulnerability of LLM-based planners, we aim to promote the development of secured robot systems. 

**Abstract (ZH)**: 机器人需要任务规划方法来实现需要多于单个动作的目标。最近，大规模语言模型（LLMs）在任务规划方面展现了令人印象深刻的表现。LLMs可以根据动作描述和目标生成逐步解决方案。尽管基于LLM的任务规划取得了成功，但对这些系统的安全方面研究有限。在本文中，我们开发了Robo-Troj，这是针对基于LLM的任务规划器的第一个多触发后门攻击，这是本文的主要贡献。作为一种多触发攻击，Robo-Troj经过训练以适应机器人类应用领域的多样性。例如，可以使用唯一的触发词“herical”来激活特定的恶意行为，如在厨房机器人上切断手。此外，我们开发了一种优化方法来选择最有效的触发词。通过展示基于LLM的规划器的脆弱性，我们旨在促进安全的机器人系统的发展。 

---
# Auditing the Ethical Logic of Generative AI Models 

**Title (ZH)**: 审核生成型人工智能模型的伦理逻辑 

**Authors**: W. Russell Neuman, Chad Coleman, Ali Dasdan, Safinah Ali, Manan Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.17544)  

**Abstract**: As generative AI models become increasingly integrated into high-stakes domains, the need for robust methods to evaluate their ethical reasoning becomes increasingly important. This paper introduces a five-dimensional audit model -- assessing Analytic Quality, Breadth of Ethical Considerations, Depth of Explanation, Consistency, and Decisiveness -- to evaluate the ethical logic of leading large language models (LLMs). Drawing on traditions from applied ethics and higher-order thinking, we present a multi-battery prompt approach, including novel ethical dilemmas, to probe the models' reasoning across diverse contexts. We benchmark seven major LLMs finding that while models generally converge on ethical decisions, they vary in explanatory rigor and moral prioritization. Chain-of-Thought prompting and reasoning-optimized models significantly enhance performance on our audit metrics. This study introduces a scalable methodology for ethical benchmarking of AI systems and highlights the potential for AI to complement human moral reasoning in complex decision-making contexts. 

**Abstract (ZH)**: 随着生成式AI模型在高风险领域中的逐步集成，评估其伦理推理的 robust 方法变得越来越重要。本文引入了一个五维度审计模型——评估分析质量、伦理考虑的广度、解释的深度、一致性和决断性，以评估领先的大语言模型（LLMs）的伦理逻辑。借鉴应用伦理学和高层次思维的传统，我们介绍了多种测试方法，包括新型伦理困境，以探索模型在多种情境下的推理能力。我们对七种主要LLMs进行了基准测试，发现虽然模型在伦理决策上普遍一致，但在解释的严谨性和道德优先级上存在差异。通过链式思考提示和推理优化模型，显著提升了我们在审计指标上的表现。本文引入了一种可扩展的AI系统伦理基准测试方法，并强调了AI在复杂决策情境中补充人类道德推理的潜力。 

---
# Towards Machine-Generated Code for the Resolution of User Intentions 

**Title (ZH)**: 面向用户意图解决的机器生成代码研究 

**Authors**: Justus Flerlage, Ilja Behnke, Odej Kao  

**Link**: [PDF](https://arxiv.org/pdf/2504.17531)  

**Abstract**: The growing capabilities of Artificial Intelligence (AI), particularly Large Language Models (LLMs), prompt a reassessment of the interaction mechanisms between users and their devices. Currently, users are required to use a set of high-level applications to achieve their desired results. However, the advent of AI may signal a shift in this regard, as its capabilities have generated novel prospects for user-provided intent resolution through the deployment of model-generated code, which is tantamount to the generation of workflows comprising a multitude of interdependent steps. This development represents a significant progression in the realm of hybrid workflows, where human and artificial intelligence collaborate to address user intentions, with the former responsible for defining these intentions and the latter for implementing the solutions to address them. In this paper, we investigate the feasibility of generating and executing workflows through code generation that results from prompting an LLM with a concrete user intention, such as \emph{Please send my car title to my insurance company}, and a simplified application programming interface for a GUI-less operating system. We provide in-depth analysis and comparison of various user intentions, the resulting code, and its execution. The findings demonstrate a general feasibility of our approach and that the employed LLM, GPT-4o-mini, exhibits remarkable proficiency in the generation of code-oriented workflows in accordance with provided user intentions. 

**Abstract (ZH)**: 人工智能（AI）尤其是大型语言模型（LLMs）能力的增强促使重新评估用户与其设备之间的交互机制。当前，用户需要使用一系列高级应用程序来达到他们的预期结果。然而，AI 的出现可能预示着这一方面的转变，因为其能力产生了通过部署模型生成代码来解决用户意图的新前景，这相当于生成由众多相互依存步骤组成的流程工作流。这一发展代表了在人类和人工智能协作解决用户意图的混合工作流领域的一个重要进步，前者负责定义这些意图，后者负责实施解决方案。在本文中，我们研究了通过提示大型语言模型生成和执行工作流代码的可行性，具体意图如“请将我的汽车登记证发送给保险公司”，并提供了一个简化版的应用程序编程接口用于无图形用户界面的操作系统。我们对各种用户意图、生成的代码及其执行进行了深入的分析和比较。研究结果表明，我们的方法具有普遍的可行性，而且所使用的语言模型GPT-4o-mini在根据提供的用户意图生成代码导向的工作流方面表现出卓越的能力。 

---
# Assessing the Capability of Large Language Models for Domain-Specific Ontology Generation 

**Title (ZH)**: 评估大型语言模型在特定领域本体生成方面的能力 

**Authors**: Anna Sofia Lippolis, Mohammad Javad Saeedizade, Robin Keskisarkka, Aldo Gangemi, Eva Blomqvist, Andrea Giovanni Nuzzolese  

**Link**: [PDF](https://arxiv.org/pdf/2504.17402)  

**Abstract**: Large Language Models (LLMs) have shown significant potential for ontology engineering. However, it is still unclear to what extent they are applicable to the task of domain-specific ontology generation. In this study, we explore the application of LLMs for automated ontology generation and evaluate their performance across different domains. Specifically, we investigate the generalizability of two state-of-the-art LLMs, DeepSeek and o1-preview, both equipped with reasoning capabilities, by generating ontologies from a set of competency questions (CQs) and related user stories. Our experimental setup comprises six distinct domains carried out in existing ontology engineering projects and a total of 95 curated CQs designed to test the models' reasoning for ontology engineering. Our findings show that with both LLMs, the performance of the experiments is remarkably consistent across all domains, indicating that these methods are capable of generalizing ontology generation tasks irrespective of the domain. These results highlight the potential of LLM-based approaches in achieving scalable and domain-agnostic ontology construction and lay the groundwork for further research into enhancing automated reasoning and knowledge representation techniques. 

**Abstract (ZH)**: 大型语言模型（LLMs）在本体工程中的应用显示了显著的潜力。然而，它们在特定领域本体生成任务中的适用程度尚不清楚。在本研究中，我们探索了LLMs在自动化本体生成中的应用，并评估了它们在不同领域中的性能。具体而言，我们通过从一组专业问题（CQs）及其相关用户故事中生成本体，研究了两种最先进的LLMs DeepSeek和o1-preview（两者均具备推理论能力）的通用性。我们的实验设计包括六个现有的本体工程项目中的不同领域，并总共使用了95个定制的专业问题，以测试模型在本体工程中的推理论证能力。研究结果表明，使用这两种LLM进行实验的效果在所有领域中都表现出惊人的一致性，表明这些方法能够跨领域泛化本体生成任务。这些发现突显了基于LLM的方法在实现可扩展且领域无关的本体构建方面的潜力，并为增强自动化推理和知识表示技术的研究奠定了基础。 

---
# AI-Enhanced Business Process Automation: A Case Study in the Insurance Domain Using Object-Centric Process Mining 

**Title (ZH)**: AI增强的企业流程自动化：基于对象中心流程挖掘在保险领域的案例研究 

**Authors**: Shahrzad Khayatbashi, Viktor Sjölind, Anders Granåker, Amin Jalali  

**Link**: [PDF](https://arxiv.org/pdf/2504.17295)  

**Abstract**: Recent advancements in Artificial Intelligence (AI), particularly Large Language Models (LLMs), have enhanced organizations' ability to reengineer business processes by automating knowledge-intensive tasks. This automation drives digital transformation, often through gradual transitions that improve process efficiency and effectiveness. To fully assess the impact of such automation, a data-driven analysis approach is needed - one that examines how traditional and AI-enhanced process variants coexist during this transition. Object-Centric Process Mining (OCPM) has emerged as a valuable method that enables such analysis, yet real-world case studies are still needed to demonstrate its applicability. This paper presents a case study from the insurance sector, where an LLM was deployed in production to automate the identification of claim parts, a task previously performed manually and identified as a bottleneck for scalability. To evaluate this transformation, we apply OCPM to assess the impact of AI-driven automation on process scalability. Our findings indicate that while LLMs significantly enhance operational capacity, they also introduce new process dynamics that require further refinement. This study also demonstrates the practical application of OCPM in a real-world setting, highlighting its advantages and limitations. 

**Abstract (ZH)**: Recent advancements in Artificial Intelligence (AI), particularly Large Language Models (LLMs), have enhanced organizations' ability to reengineer business processes by automating knowledge-intensive tasks. This automation drives digital transformation, often through gradual transitions that improve process efficiency and effectiveness. To fully assess the impact of such automation, a data-driven analysis approach is needed—one that examines how traditional and AI-enhanced process variants coexist during this transition. Object-Centric Process Mining (OCPM) has emerged as a valuable method that enables such analysis, yet real-world case studies are still needed to demonstrate its applicability. This paper presents a case study from the insurance sector, where an LLM was deployed in production to automate the identification of claim parts, a task previously performed manually and identified as a bottleneck for scalability. To evaluate this transformation, we apply OCPM to assess the impact of AI-driven automation on process scalability. Our findings indicate that while LLMs significantly enhance operational capacity, they also introduce new process dynamics that require further refinement. This study also demonstrates the practical application of OCPM in a real-world setting, highlighting its advantages and limitations. 

---
# Leveraging LLMs as Meta-Judges: A Multi-Agent Framework for Evaluating LLM Judgments 

**Title (ZH)**: 利用大规模语言模型作为元法官：一个评估大规模语言模型判断的多agent框架 

**Authors**: Yuran Li, Jama Hussein Mohamud, Chongren Sun, Di Wu, Benoit Boulet  

**Link**: [PDF](https://arxiv.org/pdf/2504.17087)  

**Abstract**: Large language models (LLMs) are being widely applied across various fields, but as tasks become more complex, evaluating their responses is increasingly challenging. Compared to human evaluators, the use of LLMs to support performance evaluation offers a more efficient alternative. However, most studies focus mainly on aligning LLMs' judgments with human preferences, overlooking the existence of biases and mistakes in human judgment. Furthermore, how to select suitable LLM judgments given multiple potential LLM responses remains underexplored. To address these two aforementioned issues, we propose a three-stage meta-judge selection pipeline: 1) developing a comprehensive rubric with GPT-4 and human experts, 2) using three advanced LLM agents to score judgments, and 3) applying a threshold to filter out low-scoring judgments. Compared to methods using a single LLM as both judge and meta-judge, our pipeline introduces multi-agent collaboration and a more comprehensive rubric. Experimental results on the JudgeBench dataset show about 15.55\% improvement compared to raw judgments and about 8.37\% improvement over the single-agent baseline. Our work demonstrates the potential of LLMs as meta-judges and lays the foundation for future research on constructing preference datasets for LLM-as-a-judge reinforcement learning. 

**Abstract (ZH)**: 大型语言模型在各领域的广泛应用使得复杂任务的评估日益具有挑战性。与人类评估者相比，使用大型语言模型辅助性能评估提供了更为高效的选择。然而，大多数研究主要关注于对齐大型语言模型的判断与人类偏好，忽略了人类判断中存在的偏见和错误。此外，如何在多种潜在的大型语言模型判断中选择合适的判断仍然缺乏探索。为解决这些问题，我们提出了一种三阶段元评估者选择流水线：1) 使用GPT-4和人类专家开发综合评分标准；2) 使用三种先进的大型语言模型代理进行评分；3) 采用阈值过滤低分判断。与使用单一大型语言模型作为评估者和元评估者的现有方法相比，我们的流水线引入了多代理协作和更为全面的评分标准。实验结果表明，与原始判断相比，改进幅度约为15.55%，与单代理基线相比，改进幅度约为8.37%。我们的工作展示了大型语言模型作为元评估者的潜力，并为未来基于大型语言模型的强化学习构建偏好数据集的研究奠定了基础。 

---
# A Desideratum for Conversational Agents: Capabilities, Challenges, and Future Directions 

**Title (ZH)**: 对话代理所需的能力、挑战与未来方向 

**Authors**: Emre Can Acikgoz, Cheng Qian, Hongru Wang, Vardhan Dongre, Xiusi Chen, Heng Ji, Dilek Hakkani-Tür, Gokhan Tur  

**Link**: [PDF](https://arxiv.org/pdf/2504.16939)  

**Abstract**: Recent advances in Large Language Models (LLMs) have propelled conversational AI from traditional dialogue systems into sophisticated agents capable of autonomous actions, contextual awareness, and multi-turn interactions with users. Yet, fundamental questions about their capabilities, limitations, and paths forward remain open. This survey paper presents a desideratum for next-generation Conversational Agents - what has been achieved, what challenges persist, and what must be done for more scalable systems that approach human-level intelligence. To that end, we systematically analyze LLM-driven Conversational Agents by organizing their capabilities into three primary dimensions: (i) Reasoning - logical, systematic thinking inspired by human intelligence for decision making, (ii) Monitor - encompassing self-awareness and user interaction monitoring, and (iii) Control - focusing on tool utilization and policy following. Building upon this, we introduce a novel taxonomy by classifying recent work on Conversational Agents around our proposed desideratum. We identify critical research gaps and outline key directions, including realistic evaluations, long-term multi-turn reasoning skills, self-evolution capabilities, collaborative and multi-agent task completion, personalization, and proactivity. This work aims to provide a structured foundation, highlight existing limitations, and offer insights into potential future research directions for Conversational Agents, ultimately advancing progress toward Artificial General Intelligence (AGI). We maintain a curated repository of papers at: this https URL. 

**Abstract (ZH)**: 近期大型语言模型的进步推动了对话AI从传统的对话系统发展为具备自主行动能力、上下文意识及多轮交互的复杂代理，但关于其能力、限制及未来路径的基本问题仍然开放。本文综述了下一代对话代理的需求——已取得的成就、现存的挑战以及为实现接近人类水平智能的更具扩展性的系统所需做的工作。为此，我们系统分析了由大型语言模型驱动的对话代理，并将其能力组织为三个主要维度：(i) 原理推理——由人类智能启发的逻辑性、系统的决策制定思维；(ii) 监控——涵盖自我意识和用户交互监控；(iii) 控制——专注于工具利用及策略遵循。在此基础上，我们提出了一个新的分类框架，通过围绕我们提出的既定需求对近期对话代理研究进行分类。我们识别了关键的研究缺口，并指出了关键的研究方向，包括现实评估、长期多轮推理能力、自我进化能力、协作和多代理任务完成、个性化及主动性。本文旨在提供一个结构化的基础，突出现有局限性，并为对话代理潜在的未来研究方向提供见解，最终推动通往通用人工智能（AGI）的进步。我们维护了一个精选的论文库：请访问此链接。 

---
# Multilingual Performance Biases of Large Language Models in Education 

**Title (ZH)**: 大型语言模型在教育中的多语言性能偏见 

**Authors**: Vansh Gupta, Sankalan Pal Chowdhury, Vilém Zouhar, Donya Rooein, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2504.17720)  

**Abstract**: Large language models (LLMs) are increasingly being adopted in educational settings. These applications expand beyond English, though current LLMs remain primarily English-centric. In this work, we ascertain if their use in education settings in non-English languages is warranted. We evaluated the performance of popular LLMs on four educational tasks: identifying student misconceptions, providing targeted feedback, interactive tutoring, and grading translations in six languages (Hindi, Arabic, Farsi, Telugu, Ukrainian, Czech) in addition to English. We find that the performance on these tasks somewhat corresponds to the amount of language represented in training data, with lower-resource languages having poorer task performance. Although the models perform reasonably well in most languages, the frequent performance drop from English is significant. Thus, we recommend that practitioners first verify that the LLM works well in the target language for their educational task before deployment. 

**Abstract (ZH)**: 大型语言模型（LLMs）在教育场景中的应用日益增多。尽管当前的LLMs主要以英语为中心，它们的应用已超出英语范围。本研究旨在确定在非英语语言环境中使用LLMs进行教育的合理性。我们评估了多个流行LLMs在六种语言（印地语、阿拉伯语、波斯语、泰卢固语、乌克兰语、捷克语）以及英语上四种教育任务的表现：识别学生误解、提供定制反馈、互动辅导以及评分翻译。我们发现，这些任务的性能与训练数据中所包含的语言量存在一定对应关系，低资源语言的任务性能较差。虽然模型在大多数语言中的表现尚可，但从英语到其他语言的频繁性能下降非常显著。因此，我们建议实践者在部署前首先验证目标语言中LLMs的表现是否符合教育任务的要求。 

---
# Ensemble Bayesian Inference: Leveraging Small Language Models to Achieve LLM-level Accuracy in Profile Matching Tasks 

**Title (ZH)**: ensemble Bayesian推断：利用小型语言模型在身份匹配任务中实现类似于大型语言模型的准确性 

**Authors**: Haru-Tada Sato, Fuka Matsuzaki, Jun-ichiro Takahashi  

**Link**: [PDF](https://arxiv.org/pdf/2504.17685)  

**Abstract**: This study explores the potential of small language model(SLM) ensembles to achieve accuracy comparable to proprietary large language models (LLMs). We propose Ensemble Bayesian Inference (EBI), a novel approach that applies Bayesian estimation to combine judgments from multiple SLMs, allowing them to exceed the performance limitations of individual models. Our experiments on diverse tasks(aptitude assessments and consumer profile analysis in both Japanese and English) demonstrate EBI's effectiveness. Notably, we analyze cases where incorporating models with negative Lift values into ensembles improves overall performance, and we examine the method's efficacy across different languages. These findings suggest new possibilities for constructing high-performance AI systems with limited computational resources and for effectively utilizing models with individually lower performance. Building on existing research on LLM performance evaluation, ensemble methods, and open-source LLM utilization, we discuss the novelty and significance of our approach. 

**Abstract (ZH)**: 本研究探讨了小型语言模型(SLM)集成在达到与专有大型语言模型(LLM)相媲美的准确性方面的潜力。我们提出了集成贝叶斯推断(EBI)，这是一种新颖的方法，通过贝叶斯估计将多个SLM的判断进行结合，使它们能够超越单一模型的性能限制。我们在日语和英语的多样化任务（如能力评估和消费者画像分析）上的实验展示了EBI的有效性。值得注意的是，我们分析了在集成中包含具有负Lift值的模型如何提高整体性能的情况，并考察了该方法在不同语言中的有效性。这些发现表明了在有限计算资源下构建高性能AI系统的新可能性，并且有效利用性能较低的模型的新方法。基于现有对LLM性能评估、集成方法以及开源LLM利用的研究，我们讨论了我们方法的新颖性和重要性。 

---
# INSIGHT: Bridging the Student-Teacher Gap in Times of Large Language Models 

**Title (ZH)**: INSIGHT: 缩减大规模语言模型时代的学生与教师差距 

**Authors**: Jarne Thys, Sebe Vanbrabant, Davy Vanacken, Gustavo Rovelo Ruiz  

**Link**: [PDF](https://arxiv.org/pdf/2504.17677)  

**Abstract**: The rise of AI, especially Large Language Models, presents challenges and opportunities to integrate such technology into the classroom. AI has the potential to revolutionize education by helping teaching staff with various tasks, such as personalizing their teaching methods, but it also raises concerns, for example, about the degradation of student-teacher interactions and user privacy. This paper introduces INSIGHT, a proof of concept to combine various AI tools to assist teaching staff and students in the process of solving exercises. INSIGHT has a modular design that allows it to be integrated into various higher education courses. We analyze students' questions to an LLM by extracting keywords, which we use to dynamically build an FAQ from students' questions and provide new insights for the teaching staff to use for more personalized face-to-face support. Future work could build upon INSIGHT by using the collected data to provide adaptive learning and adjust content based on student progress and learning styles to offer a more interactive and inclusive learning experience. 

**Abstract (ZH)**: AI的兴起，尤其是大型语言模型，为将此类技术融入课堂带来了挑战与机遇。AI有潜力通过帮助教学人员完成各种任务来革新教育，例如个性化教学方法，但也引发了关于学生-教师互动质量下降和用户隐私的问题。本文介绍了INSIGHT，一种概念验证工具，旨在结合多种AI工具以辅助教师和学生在解决练习题过程中。INSIGHT采用模块化设计，可集成到各种高等教育课程中。通过对学生问题的关键词进行提取，动态构建FAQ，并为教师提供新的见解，以支持更具个性化的面对面支持。未来的工作可以通过利用收集的数据来提供自适应学习，并根据学生的学习进展和学习风格调整内容，以提供更互动和包容的学习体验。 

---
# HalluLens: LLM Hallucination Benchmark 

**Title (ZH)**: HalluLens: LLM Hallucination Benchmark 

**Authors**: Yejin Bang, Ziwei Ji, Alan Schelten, Anthony Hartshorn, Tara Fowler, Cheng Zhang, Nicola Cancedda, Pascale Fung  

**Link**: [PDF](https://arxiv.org/pdf/2504.17550)  

**Abstract**: Large language models (LLMs) often generate responses that deviate from user input or training data, a phenomenon known as "hallucination." These hallucinations undermine user trust and hinder the adoption of generative AI systems. Addressing hallucinations is essential for the advancement of LLMs. This paper introduces a comprehensive hallucination benchmark, incorporating both new extrinsic and existing intrinsic evaluation tasks, built upon clear taxonomy of hallucination. A major challenge in benchmarking hallucinations is the lack of a unified framework due to inconsistent definitions and categorizations. We disentangle LLM hallucination from "factuality," proposing a clear taxonomy that distinguishes between extrinsic and intrinsic hallucinations, to promote consistency and facilitate research. Extrinsic hallucinations, where the generated content is not consistent with the training data, are increasingly important as LLMs evolve. Our benchmark includes dynamic test set generation to mitigate data leakage and ensure robustness against such leakage. We also analyze existing benchmarks, highlighting their limitations and saturation. The work aims to: (1) establish a clear taxonomy of hallucinations, (2) introduce new extrinsic hallucination tasks, with data that can be dynamically regenerated to prevent saturation by leakage, (3) provide a comprehensive analysis of existing benchmarks, distinguishing them from factuality evaluations. 

**Abstract (ZH)**: 大型语言模型（LLMs）常常生成与用户输入或训练数据不符的响应，这一现象被称为“幻觉”。这些幻觉削弱了用户信任并阻碍生成式AI系统的应用。解决幻觉问题对于LLM的发展至关重要。本文介绍了一个全面的幻觉基准，包含新的外部评价任务和现有的内部评价任务，并基于清晰的幻觉分类。基准测试的主要挑战在于缺乏统一框架，这源于不一致的定义和分类。我们从“事实性”中分离出LLM的幻觉，并提出了一种清晰的分类体系，区分外部和内部幻觉，以促进一致性并促进研究。外部幻觉是指生成的内容不与训练数据一致，随着LLM的发展变得越来越重要。我们的基准测试包括动态测试集生成以减轻数据泄漏，并确保在数据泄漏方面的稳健性。我们还分析了现有基准测试，指出了其局限性和饱和度。本项工作旨在：（1）建立清晰的幻觉分类体系；（2）引入新的外部幻觉任务，数据可以动态再生以防止泄漏导致的饱和；（3）对现有基准测试进行全面分析，区分其与事实性评估的差异。 

---
# Plasticine: Accelerating Research in Plasticity-Motivated Deep Reinforcement Learning 

**Title (ZH)**: Plasticine: 加速源自塑性动机的深度强化学习研究 

**Authors**: Mingqi Yuan, Qi Wang, Guozheng Ma, Bo Li, Xin Jin, Yunbo Wang, Xiaokang Yang, Wenjun Zeng, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2504.17490)  

**Abstract**: Developing lifelong learning agents is crucial for artificial general intelligence. However, deep reinforcement learning (RL) systems often suffer from plasticity loss, where neural networks gradually lose their ability to adapt during training. Despite its significance, this field lacks unified benchmarks and evaluation protocols. We introduce Plasticine, the first open-source framework for benchmarking plasticity optimization in deep RL. Plasticine provides single-file implementations of over 13 mitigation methods, 10 evaluation metrics, and learning scenarios with increasing non-stationarity levels from standard to open-ended environments. This framework enables researchers to systematically quantify plasticity loss, evaluate mitigation strategies, and analyze plasticity dynamics across different contexts. Our documentation, examples, and source code are available at this https URL. 

**Abstract (ZH)**: 开发终身学习代理对于通用人工智能至关重要。然而，深度强化学习（RL）系统常常会遭受塑性丧失的问题，即神经网络在训练过程中逐渐失去其适应能力。尽管这个问题很重要，但该领域缺乏统一的基准和评估协议。我们引入了Plasticine，这是首个开源框架，用于评估深度RL中的塑性优化。Plasticine提供了超过13种减轻方法、10种评估指标以及从标准环境到开放环境的不断增加非稳态性的学习场景的单文件实现。该框架使研究人员能够系统地量化塑性丧失、评估减轻策略，并分析在不同上下文中的塑性动态。我们的文档、示例和源代码可在以下链接获取。 

---
# HMI: Hierarchical Knowledge Management for Efficient Multi-Tenant Inference in Pretrained Language Models 

**Title (ZH)**: HMI：层次化知识管理以实现高效预训练语言模型多租户推理 

**Authors**: Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, Qin Xie, Guiming Xie, Xuejian Gong  

**Link**: [PDF](https://arxiv.org/pdf/2504.17449)  

**Abstract**: The significant computational demands of pretrained language models (PLMs), which often require dedicated hardware, present a substantial challenge in serving them efficiently, especially in multi-tenant environments. To address this, we introduce HMI, a Hierarchical knowledge management-based Multi-tenant Inference system, designed to manage tenants with distinct PLMs resource-efficiently. Our approach is three-fold: Firstly, we categorize PLM knowledge into general, domain-specific, and task-specific. Leveraging insights on knowledge acquisition across different model layers, we construct hierarchical PLMs (hPLMs) by extracting and storing knowledge at different levels, significantly reducing GPU memory usage per tenant. Secondly, we establish hierarchical knowledge management for hPLMs generated by various tenants in HMI. We manage domain-specific knowledge with acceptable storage increases by constructing and updating domain-specific knowledge trees based on frequency. We manage task-specific knowledge within limited GPU memory through parameter swapping. Finally, we propose system optimizations to enhance resource utilization and inference throughput. These include fine-grained pipelining via hierarchical knowledge prefetching to overlap CPU and I/O operations with GPU computations, and optimizing parallel implementations with batched matrix multiplications. Our experimental results demonstrate that the proposed HMI can efficiently serve up to 10,000 hPLMs (hBERTs and hGPTs) on a single GPU, with only a negligible compromise in accuracy. 

**Abstract (ZH)**: 基于层次知识管理的多租户推理系统HMI：资源高效服务于预训练语言模型 

---
# Towards Leveraging Large Language Model Summaries for Topic Modeling in Source Code 

**Title (ZH)**: 基于大型语言模型摘要的主题建模在源代码中的应用探索 

**Authors**: Michele Carissimi, Martina Saletta, Claudio Ferretti  

**Link**: [PDF](https://arxiv.org/pdf/2504.17426)  

**Abstract**: Understanding source code is a topic of great interest in the software engineering community, since it can help programmers in various tasks such as software maintenance and reuse. Recent advances in large language models (LLMs) have demonstrated remarkable program comprehension capabilities, while transformer-based topic modeling techniques offer effective ways to extract semantic information from text. This paper proposes and explores a novel approach that combines these strengths to automatically identify meaningful topics in a corpus of Python programs. Our method consists in applying topic modeling on the descriptions obtained by asking an LLM to summarize the code. To assess the internal consistency of the extracted topics, we compare them against topics inferred from function names alone, and those derived from existing docstrings. Experimental results suggest that leveraging LLM-generated summaries provides interpretable and semantically rich representation of code structure. The promising results suggest that our approach can be fruitfully applied in various software engineering tasks such as automatic documentation and tagging, code search, software reorganization and knowledge discovery in large repositories. 

**Abstract (ZH)**: 理解源代码是软件工程领域的一个重要课题，因为它有助于程序员在软件维护和重用等多种任务中。近期大型语言模型（LLMs）的发展展示了其出色的程序理解能力，而基于变换器的主题建模技术提供了从文本中提取语义信息的有效方法。本文提出并探索了一种结合这些优点的新方法，以自动识别代码汇集中有意义的主题。该方法是在请求LLM总结代码后应用主题建模。为了评估提取主题的一致性，我们将它们与仅从函数名称和现有文档字符串中推断的主题进行比较。实验结果表明，利用LLM生成的摘要提供了代码结构的可解释且语义丰富的表示。这些有希望的结果表明，我们的方法可以在诸如自动文档生成和标记、代码搜索、软件重组和大型代码库中的知识发现等软件工程任务中得到广泛应用。 

---
# Towards Harnessing the Collaborative Power of Large and Small Models for Domain Tasks 

**Title (ZH)**: 面向领域任务利用大型和小型模型的协作力量 

**Authors**: Yang Liu, Bingjie Yan, Tianyuan Zou, Jianqing Zhang, Zixuan Gu, Jianbing Ding, Xidong Wang, Jingyi Li, Xiaozhou Ye, Ye Ouyang, Qiang Yang, Ya-Qin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.17421)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities, but they require vast amounts of data and computational resources. In contrast, smaller models (SMs), while less powerful, can be more efficient and tailored to specific domains. In this position paper, we argue that taking a collaborative approach, where large and small models work synergistically, can accelerate the adaptation of LLMs to private domains and unlock new potential in AI. We explore various strategies for model collaboration and identify potential challenges and opportunities. Building upon this, we advocate for industry-driven research that prioritizes multi-objective benchmarks on real-world private datasets and applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）展现了卓越的能力，但需要大量的数据和计算资源。相比之下，较小的模型（SMs）虽然能力较弱，但在效率和特定领域定制方面更具优势。在这篇立场论文中，我们论证了采取协作方法，使大型和小型模型协同工作，可以加速LLMs在私有领域的适应，并解锁新的AI潜力。我们探讨了各种模型协作策略，识别潜在的挑战与机遇，并呼吁以行业驱动的研究，优先在真实世界的私有数据集和应用上建立多目标基准。 

---
# LiveLongBench: Tackling Long-Context Understanding for Spoken Texts from Live Streams 

**Title (ZH)**: LongLongBench: 应对直播流中长上下文理解的 spoken 文本verständnis 

**Authors**: Yongxuan Wu, Runyu Chen, Peiyu Liu, Hongjin Qian  

**Link**: [PDF](https://arxiv.org/pdf/2504.17366)  

**Abstract**: Long-context understanding poses significant challenges in natural language processing, particularly for real-world dialogues characterized by speech-based elements, high redundancy, and uneven information density. Although large language models (LLMs) achieve impressive results on existing benchmarks, these datasets fail to reflect the complexities of such texts, limiting their applicability to practical scenarios. To bridge this gap, we construct the first spoken long-text dataset, derived from live streams, designed to reflect the redundancy-rich and conversational nature of real-world scenarios. We construct tasks in three categories: retrieval-dependent, reasoning-dependent, and hybrid. We then evaluate both popular LLMs and specialized methods to assess their ability to understand long-contexts in these tasks. Our results show that current methods exhibit strong task-specific preferences and perform poorly on highly redundant inputs, with no single method consistently outperforming others. We propose a new baseline that better handles redundancy in spoken text and achieves strong performance across tasks. Our findings highlight key limitations of current methods and suggest future directions for improving long-context understanding. Finally, our benchmark fills a gap in evaluating long-context spoken language understanding and provides a practical foundation for developing real-world e-commerce systems. The code and benchmark are available at this https URL. 

**Abstract (ZH)**: 长上下文理解在自然语言处理中面临显著挑战，特别是在以口语元素为特征、高冗余度和不均匀信息密度为特点的实际对话场景中。尽管大规模语言模型（LLMs）在现有基准测试中取得了令人印象深刻的成果，但这些数据集未能反映此类文本的复杂性，限制了其在实际场景中的应用。为了弥合这一差距，我们构建了首个基于直播流的口语长文本数据集，旨在反映出实际场景中的高冗余和对话性质。我们构建了三类任务：检索依赖型、推理依赖型和混合型。然后，我们评估了流行的LLMs和专门方法，以评估它们在这些任务中理解长上下文的能力。我们的结果显示，当前方法表现出强烈的任务特异性偏好，并且在高度冗余的输入上表现不佳，没有一种方法在所有任务中始终优于其他方法。我们提出了一种新的基线，更好地处理口语文本中的冗余，并在所有任务中取得了优异的性能。我们的发现揭示了当前方法的关键局限性，并提出了改进长上下文理解的未来方向。最后，我们的基准填补了评估长上下文口语语言理解的空白，并为开发实际电子商务系统提供了实用的基础。相关代码和基准可在此处访问：this https URL。 

---
# Exploring Context-aware and LLM-driven Locomotion for Immersive Virtual Reality 

**Title (ZH)**: 基于情境感知和大语言模型驱动的沉浸式虚拟现实运动探索 

**Authors**: Süleyman Özdel, Kadir Burak Buldu, Enkelejda Kasneci, Efe Bozkir  

**Link**: [PDF](https://arxiv.org/pdf/2504.17331)  

**Abstract**: Locomotion plays a crucial role in shaping the user experience within virtual reality environments. In particular, hands-free locomotion offers a valuable alternative by supporting accessibility and freeing users from reliance on handheld controllers. To this end, traditional speech-based methods often depend on rigid command sets, limiting the naturalness and flexibility of interaction. In this study, we propose a novel locomotion technique powered by large language models (LLMs), which allows users to navigate virtual environments using natural language with contextual awareness. We evaluate three locomotion methods: controller-based teleportation, voice-based steering, and our language model-driven approach. Our evaluation measures include eye-tracking data analysis, including explainable machine learning through SHAP analysis as well as standardized questionnaires for usability, presence, cybersickness, and cognitive load to examine user attention and engagement. Our findings indicate that the LLM-driven locomotion possesses comparable usability, presence, and cybersickness scores to established methods like teleportation, demonstrating its novel potential as a comfortable, natural language-based, hands-free alternative. In addition, it enhances user attention within the virtual environment, suggesting greater engagement. Complementary to these findings, SHAP analysis revealed that fixation, saccade, and pupil-related features vary across techniques, indicating distinct patterns of visual attention and cognitive processing. Overall, we state that our method can facilitate hands-free locomotion in virtual spaces, especially in supporting accessibility. 

**Abstract (ZH)**: 基于大规模语言模型的自然语言驱动导航技术在虚拟现实环境中的应用研究 

---
# FLUKE: A Linguistically-Driven and Task-Agnostic Framework for Robustness Evaluation 

**Title (ZH)**: FLUKE：一种基于语言且任务无关的鲁棒性评估框架 

**Authors**: Yulia Otmakhova, Hung Thinh Truong, Rahmad Mahendra, Zenan Zhai, Rongxin Zhu, Daniel Beck, Jey Han Lau  

**Link**: [PDF](https://arxiv.org/pdf/2504.17311)  

**Abstract**: We present FLUKE (Framework for LingUistically-driven and tasK-agnostic robustness Evaluation), a task-agnostic framework for assessing model robustness through systematic minimal variations of test data. FLUKE introduces controlled variations across linguistic levels - from orthography to dialect and style varieties - and leverages large language models (LLMs) with human validation to generate modifications. We demonstrate FLUKE's utility by evaluating both fine-tuned models and LLMs across four diverse NLP tasks, and reveal that (1) the impact of linguistic variations is highly task-dependent, with some tests being critical for certain tasks but irrelevant for others; (2) while LLMs have better overall robustness compared to fine-tuned models, they still exhibit significant brittleness to certain linguistic variations; (3) all models show substantial vulnerability to negation modifications across most tasks. These findings highlight the importance of systematic robustness testing for understanding model behaviors. 

**Abstract (ZH)**: 我们提出了FLUKE（基于语言驱动和任务无关稳健性评估框架），一个通过系统地对测试数据进行最小变化来评估模型 robustness 的任务无关框架。FLUKE 在语言层面引入了可控变化——从拼写到方言和风格变体，并利用大型语言模型（LLMs）结合人工验证来生成修改。我们通过在四个不同的 NLP 任务上评估调优模型和 LLMs，展示了 FLUKE 的实用性，并揭示了以下几点：(1) 语言变化的影响高度依赖于任务，某些测试对某些任务至关重要，但对其他任务则无关紧要；(2) 尽管 LLMs 具有比调优模型更好的整体稳健性，但在某些语言变化方面仍表现出显著的脆弱性；(3) 所有模型在大多数任务中都对否定修饰表现出明显的脆弱性。这些发现强调了系统稳健性测试对于理解模型行为的重要性。 

---
# You Are What You Bought: Generating Customer Personas for E-commerce Applications 

**Title (ZH)**: 你是你所购买的：为电子商务应用生成客户画像 

**Authors**: Yimin Shi, Yang Fei, Shiqi Zhang, Haixun Wang, Xiaokui Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.17304)  

**Abstract**: In e-commerce, user representations are essential for various applications. Existing methods often use deep learning techniques to convert customer behaviors into implicit embeddings. However, these embeddings are difficult to understand and integrate with external knowledge, limiting the effectiveness of applications such as customer segmentation, search navigation, and product recommendations. To address this, our paper introduces the concept of the customer persona. Condensed from a customer's numerous purchasing histories, a customer persona provides a multi-faceted and human-readable characterization of specific purchase behaviors and preferences, such as Busy Parents or Bargain Hunters.
This work then focuses on representing each customer by multiple personas from a predefined set, achieving readable and informative explicit user representations. To this end, we propose an effective and efficient solution GPLR. To ensure effectiveness, GPLR leverages pre-trained LLMs to infer personas for customers. To reduce overhead, GPLR applies LLM-based labeling to only a fraction of users and utilizes a random walk technique to predict personas for the remaining customers. We further propose RevAff, which provides an absolute error $\epsilon$ guarantee while improving the time complexity of the exact solution by a factor of at least $O(\frac{\epsilon\cdot|E|N}{|E|+N\log N})$, where $N$ represents the number of customers and products, and $E$ represents the interactions between them. We evaluate the performance of our persona-based representation in terms of accuracy and robustness for recommendation and customer segmentation tasks using three real-world e-commerce datasets. Most notably, we find that integrating customer persona representations improves the state-of-the-art graph convolution-based recommendation model by up to 12% in terms of NDCG@K and F1-Score@K. 

**Abstract (ZH)**: 电子商务中，用户表示对于各种应用至关重要。现有的方法往往使用深度学习技术将客户行为转换为隐式嵌入。然而，这些嵌入难以理解和与外部知识集成，限制了客户细分、搜索导航和产品推荐等应用程序的效果。为解决这一问题，本文引入了客户人像的概念。从客户的众多购买历史中提炼出客户人像，提供了一种多维度和人性化的特定购买行为和偏好表征，如忙碌的父母或精明的购物者。

本文随后关注通过预定义集中的多个人像表示每位客户，实现人性化的明确用户表示。为此，我们提出了一种有效且高效的解决方案GPLR。为确保有效性，GPLR利用预训练的语言模型推断客户的各种人像。为减少开销，GPLR仅对一部分用户应用基于语言模型的标签，并使用随机漫步技术预测其余客户的各种人像。此外，我们提出了RevAff，它在提供绝对误差$\epsilon$保证的同时，将精确解的时间复杂度至少提高了$O(\frac{\epsilon\cdot|E|N}{|E|+N\log N})$倍，其中$N$表示客户和产品的数量，$E$表示它们之间的交互。我们使用三个实际电商数据集评估基于人像的表示在推荐和客户细分任务中的准确性和鲁棒性。值得注意的是，我们发现集成客户人像表示提高了基于图卷积的最新推荐模型在NDCG@K和F1-Score@K上的效果，最多可提升12%。 

---
# Automatically Generating Rules of Malicious Software Packages via Large Language Model 

**Title (ZH)**: 通过大型语言模型自动生成恶意软件包的规则 

**Authors**: XiangRui Zhang, HaoYu Chen, Yongzhong He, Wenjia Niu, Qiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.17198)  

**Abstract**: Today's security tools predominantly rely on predefined rules crafted by experts, making them poorly adapted to the emergence of software supply chain attacks. To tackle this limitation, we propose a novel tool, RuleLLM, which leverages large language models (LLMs) to automate rule generation for OSS ecosystems. RuleLLM extracts metadata and code snippets from malware as its input, producing YARA and Semgrep rules that can be directly deployed in software development. Specifically, the rule generation task involves three subtasks: crafting rules, refining rules, and aligning rules. To validate RuleLLM's effectiveness, we implemented a prototype system and conducted experiments on the dataset of 1,633 malicious packages. The results are promising that RuleLLM generated 763 rules (452 YARA and 311 Semgrep) with a precision of 85.2\% and a recall of 91.8\%, outperforming state-of-the-art (SOTA) tools and scored-based approaches. We further analyzed generated rules and proposed a rule taxonomy: 11 categories and 38 subcategories. 

**Abstract (ZH)**: 当前的安全工具主要依赖于专家制定的预定义规则，这使它们在应对软件供应链攻击时显得不够灵活。为了应对这一局限性，我们提出了一种新型工具RuleLLM，该工具利用大规模语言模型（LLMs）自动为开源软件生态系统生成规则。RuleLLM 以恶意软件的元数据和代码片段作为输入，生成可以直接部署在软件开发中的YARA和Semgrep规则。具体来说，规则生成任务包括三个子任务：制定规则、优化规则和对齐规则。为了验证RuleLLM的有效性，我们实现了一个原型系统，并在包含1,633个恶意包的数据集上进行了实验。结果显示，RuleLLM生成了763条规则（其中452条为YARA规则，311条为Semgrep规则），精确率为85.2%，召回率为91.8%，在性能上超越了当前最先进的工具和基于评分的方法。我们进一步分析了生成的规则，并提出了一个规则分类体系：11个大类和38个子类。 

---
# MIRAGE: A Metric-Intensive Benchmark for Retrieval-Augmented Generation Evaluation 

**Title (ZH)**: MIRAGE：一个基于度量的标准，用于检索增强生成评估 

**Authors**: Chanhee Park, Hyeonseok Moon, Chanjun Park, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2504.17137)  

**Abstract**: Retrieval-Augmented Generation (RAG) has gained prominence as an effective method for enhancing the generative capabilities of Large Language Models (LLMs) through the incorporation of external knowledge. However, the evaluation of RAG systems remains a challenge, due to the intricate interplay between retrieval and generation components. This limitation has resulted in a scarcity of benchmarks that facilitate a detailed, component-specific assessment. In this work, we present MIRAGE, a Question Answering dataset specifically designed for RAG evaluation. MIRAGE consists of 7,560 curated instances mapped to a retrieval pool of 37,800 entries, enabling an efficient and precise evaluation of both retrieval and generation tasks. We also introduce novel evaluation metrics aimed at measuring RAG adaptability, encompassing dimensions such as noise vulnerability, context acceptability, context insensitivity, and context misinterpretation. Through comprehensive experiments across various retriever-LLM configurations, we provide new insights into the optimal alignment of model pairs and the nuanced dynamics within RAG systems. The dataset and evaluation code are publicly available, allowing for seamless integration and customization in diverse research settings\footnote{The MIRAGE code and data are available at this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）作为一种通过融入外部知识来提升大型语言模型（LLMs）生成能力的有效方法已经受到了广泛关注。然而，由于检索和生成组件之间的复杂交互，RAG系统的评估仍然具有挑战性。这一局限性导致了详细的、组件特定的评估基准的缺乏。在此工作中，我们提出了MIRAGE，一个专门用于RAG评估的问答数据集。MIRAGE包含7,560个精心策划的实例，映射到一个包含37,800条条目的检索池中，使检索和生成任务的高效和精确评估成为可能。我们还引入了新的评估指标，旨在衡量RAG的适应性，包括噪声脆弱性、上下文接受性、上下文无关性和上下文误读等多个维度。通过各种检索-LLM配置下的全面实验，我们提供了模型对齐和RAG系统内部微妙动态的新见解。数据集和评估代码已公开，可在各种研究环境中轻松集成和定制。 

---
# The Rise of Small Language Models in Healthcare: A Comprehensive Survey 

**Title (ZH)**: 小语言模型在医疗健康领域的兴起：一项综合调研 

**Authors**: Muskan Garg, Shaina Raza, Shebuti Rayana, Xingyi Liu, Sunghwan Sohn  

**Link**: [PDF](https://arxiv.org/pdf/2504.17119)  

**Abstract**: Despite substantial progress in healthcare applications driven by large language models (LLMs), growing concerns around data privacy, and limited resources; the small language models (SLMs) offer a scalable and clinically viable solution for efficient performance in resource-constrained environments for next-generation healthcare informatics. Our comprehensive survey presents a taxonomic framework to identify and categorize them for healthcare professionals and informaticians. The timeline of healthcare SLM contributions establishes a foundational framework for analyzing models across three dimensions: NLP tasks, stakeholder roles, and the continuum of care. We present a taxonomic framework to identify the architectural foundations for building models from scratch; adapting SLMs to clinical precision through prompting, instruction fine-tuning, and reasoning; and accessibility and sustainability through compression techniques. Our primary objective is to offer a comprehensive survey for healthcare professionals, introducing recent innovations in model optimization and equipping them with curated resources to support future research and development in the field. Aiming to showcase the groundbreaking advancements in SLMs for healthcare, we present a comprehensive compilation of experimental results across widely studied NLP tasks in healthcare to highlight the transformative potential of SLMs in healthcare. The updated repository is available at Github 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在医疗保健应用方面取得了显著进展，但随着对数据隐私的关注不断增加和资源有限；小型语言模型（SLMs）提供了在资源受限环境中实现高效性能的可扩展且临床实用的解决方案，适用于下一代医疗informatics。我们的综合综述提出了一种分类框架，以识别和分类SLMs，供医疗专业人员和informatics专家使用。医疗保健SLMs的历史时间轴建立了一个基础框架，用于从三个维度分析模型：NLP任务、利益相关者角色和照护连续性。我们提出了一种分类框架，以识别构建模型的基础架构；通过提示、指令微调和推理将SLMs适应临床精度；并通过压缩技术实现可访问性和可持续性。我们的主要目标是为医疗保健专业人员提供一个全面的综述，介绍模型优化的最新创新，并为他们提供经过筛选的资源，以支持未来该领域的研究和开发。旨在展示医疗保健领域小型语言模型（SLMs）的突破性进步，我们提供了一个广泛的自然语言处理（NLP）任务实验结果的综合汇总，以突出SLMs在医疗保健领域的变革潜力。更新后的仓库在Github上可用。 

---
# DyMU: Dynamic Merging and Virtual Unmerging for Efficient VLMs 

**Title (ZH)**: DyMU: 动态合并与虚拟拆分以实现高效的大模型 

**Authors**: Zhenhailong Wang, Senthil Purushwalkam, Caiming Xiong, Silvio Savarese, Heng Ji, Ran Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.17040)  

**Abstract**: We present DyMU, an efficient, training-free framework that dynamically reduces the computational burden of vision-language models (VLMs) while maintaining high task performance. Our approach comprises two key components. First, Dynamic Token Merging (DToMe) reduces the number of visual token embeddings by merging similar tokens based on image complexity, addressing the inherent inefficiency of fixed-length outputs in vision transformers. Second, Virtual Token Unmerging (VTU) simulates the expected token sequence for large language models (LLMs) by efficiently reconstructing the attention dynamics of a full sequence, thus preserving the downstream performance without additional fine-tuning. Unlike previous approaches, our method dynamically adapts token compression to the content of the image and operates completely training-free, making it readily applicable to most state-of-the-art VLM architectures. Extensive experiments on image and video understanding tasks demonstrate that DyMU can reduce the average visual token count by 32%-85% while achieving comparable performance to full-length models across diverse VLM architectures, including the recently popularized AnyRes-based visual encoders. Furthermore, through qualitative analyses, we demonstrate that DToMe effectively adapts token reduction based on image complexity and, unlike existing systems, provides users more control over computational costs. Project page: this https URL. 

**Abstract (ZH)**: 我们提出了DyMU，一种高效且无需训练的框架，可在保持高任务性能的同时动态减少视觉语言模型（VLMs）的计算负担。 

---
# (Im)possibility of Automated Hallucination Detection in Large Language Models 

**Title (ZH)**: 大型语言模型中自动化幻觉检测的可能性研究（或：大型语言模型中自动化幻觉检测的不可能性研究） 

**Authors**: Amin Karbasi, Omar Montasser, John Sous, Grigoris Velegkas  

**Link**: [PDF](https://arxiv.org/pdf/2504.17004)  

**Abstract**: Is automated hallucination detection possible? In this work, we introduce a theoretical framework to analyze the feasibility of automatically detecting hallucinations produced by large language models (LLMs). Inspired by the classical Gold-Angluin framework for language identification and its recent adaptation to language generation by Kleinberg and Mullainathan, we investigate whether an algorithm, trained on examples drawn from an unknown target language $K$ (selected from a countable collection) and given access to an LLM, can reliably determine whether the LLM's outputs are correct or constitute hallucinations.
First, we establish an equivalence between hallucination detection and the classical task of language identification. We prove that any hallucination detection method can be converted into a language identification method, and conversely, algorithms solving language identification can be adapted for hallucination detection. Given the inherent difficulty of language identification, this implies that hallucination detection is fundamentally impossible for most language collections if the detector is trained using only correct examples from the target language.
Second, we show that the use of expert-labeled feedback, i.e., training the detector with both positive examples (correct statements) and negative examples (explicitly labeled incorrect statements), dramatically changes this conclusion. Under this enriched training regime, automated hallucination detection becomes possible for all countable language collections.
These results highlight the essential role of expert-labeled examples in training hallucination detectors and provide theoretical support for feedback-based methods, such as reinforcement learning with human feedback (RLHF), which have proven critical for reliable LLM deployment. 

**Abstract (ZH)**: 自动幻觉检测的可能性：理论框架探究 

---
# Backslash: Rate Constrained Optimized Training of Large Language Models 

**Title (ZH)**: Backslash: 大型语言模型在速率约束下的优化训练 

**Authors**: Jun Wu, Jiangtao Wen, Yuxing Han  

**Link**: [PDF](https://arxiv.org/pdf/2504.16968)  

**Abstract**: The rapid advancement of large-language models (LLMs) has driven extensive research into parameter compression after training has been completed, yet compression during the training phase remains largely unexplored. In this work, we introduce Rate-Constrained Training (Backslash), a novel training-time compression approach based on rate-distortion optimization (RDO). Backslash enables a flexible trade-off between model accuracy and complexity, significantly reducing parameter redundancy while preserving performance. Experiments in various architectures and tasks demonstrate that Backslash can reduce memory usage by 60\% - 90\% without accuracy loss and provides significant compression gain compared to compression after training. Moreover, Backslash proves to be highly versatile: it enhances generalization with small Lagrange multipliers, improves model robustness to pruning (maintaining accuracy even at 80\% pruning rates), and enables network simplification for accelerated inference on edge devices. 

**Abstract (ZH)**: 大规模语言模型的快速进步推动了训练完成后参数压缩的广泛应用，然而训练期压缩仍然鲜有探索。本文介绍了一种基于率失真优化的新训练时压缩方法——Backslash。Backslash能够在模型准确性与复杂性之间灵活权衡，显著减少参数冗余同时保持性能。在多种架构和任务上的实验显示，Backslash能够在不损失准确性的前提下将内存使用量减少60%至90%，并且相比训练后压缩提供了显著的压缩增益。此外，Backslash具有高度的灵活性：使用小的拉格朗日乘数增强泛化能力、提高模型修剪的稳健性（在80%的修剪率下仍保持准确性），并能够简化网络以加速边缘设备上的推理。 

---
# Intrinsic Barriers to Explaining Deep Foundation Models 

**Title (ZH)**: 深度基础模型内在的解释障碍 

**Authors**: Zhen Tan, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.16948)  

**Abstract**: Deep Foundation Models (DFMs) offer unprecedented capabilities but their increasing complexity presents profound challenges to understanding their internal workings-a critical need for ensuring trust, safety, and accountability. As we grapple with explaining these systems, a fundamental question emerges: Are the difficulties we face merely temporary hurdles, awaiting more sophisticated analytical techniques, or do they stem from \emph{intrinsic barriers} deeply rooted in the nature of these large-scale models themselves? This paper delves into this critical question by examining the fundamental characteristics of DFMs and scrutinizing the limitations encountered by current explainability methods when confronted with this inherent challenge. We probe the feasibility of achieving satisfactory explanations and consider the implications for how we must approach the verification and governance of these powerful technologies. 

**Abstract (ZH)**: Deep Foundation Models (DFMs)提供了前所未有的能力，但其日益复杂的结构对理解其内部工作机制提出了深刻的挑战——这对确保信任、安全和问责至关重要。在我们努力解释这些系统时，一个基本问题浮出水面：我们面临的困难只是等待更先进的分析技术的暂时障碍，还是源于这些大规模模型本身内在的根本限制？本文通过探讨DFMs的基本特征，并审视当前可解释性方法在面对这一固有挑战时遇到的局限性，来探索这一关键问题。我们探究获得满意解释的可行性，并考虑这对如何验证和治理这些强大技术的影响。 

---
