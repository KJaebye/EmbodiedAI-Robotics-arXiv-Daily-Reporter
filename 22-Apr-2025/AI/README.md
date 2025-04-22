# Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning 

**Title (ZH)**: 停止求和：最小形式的信用分配即为推理过程奖励模型所需的一切。 

**Authors**: Jie Cheng, Ruixi Qiao, Lijun Li, Chao Guo, Junle Wang, Gang Xiong, Yisheng Lv, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15275)  

**Abstract**: Process reward models (PRMs) have proven effective for test-time scaling of Large Language Models (LLMs) on challenging reasoning tasks. However, reward hacking issues with PRMs limit their successful application in reinforcement fine-tuning. In this paper, we identify the main cause of PRM-induced reward hacking: the canonical summation-form credit assignment in reinforcement learning (RL), which defines the value as cumulative gamma-decayed future rewards, easily induces LLMs to hack steps with high rewards. To address this, we propose PURE: Process sUpervised Reinforcement lEarning. The key innovation of PURE is a min-form credit assignment that formulates the value function as the minimum of future rewards. This method significantly alleviates reward hacking by limiting the value function range and distributing advantages more reasonably. Through extensive experiments on 3 base models, we show that PRM-based approaches enabling min-form credit assignment achieve comparable reasoning performance to verifiable reward-based methods within only 30% steps. In contrast, the canonical sum-form credit assignment collapses training even at the beginning! Additionally, when we supplement PRM-based fine-tuning with just 10% verifiable rewards, we further alleviate reward hacking and produce the best fine-tuned model based on Qwen2.5-Math-7B in our experiments, achieving 82.5% accuracy on AMC23 and 53.3% average accuracy across 5 benchmarks. Moreover, we summarize the observed reward hacking cases and analyze the causes of training collapse. Code and models are available at this https URL. 

**Abstract (ZH)**: 过程奖励模型诱导的奖励作弊问题及其解决方法：过程监督强化学习（PURE） 

---
# Leveraging Language Models for Automated Patient Record Linkage 

**Title (ZH)**: 利用语言模型进行自动化患者记录链接 

**Authors**: Mohammad Beheshti, Lovedeep Gondara, Iris Zachary  

**Link**: [PDF](https://arxiv.org/pdf/2504.15261)  

**Abstract**: Objective: Healthcare data fragmentation presents a major challenge for linking patient data, necessitating robust record linkage to integrate patient records from diverse sources. This study investigates the feasibility of leveraging language models for automated patient record linkage, focusing on two key tasks: blocking and matching. Materials and Methods: We utilized real-world healthcare data from the Missouri Cancer Registry and Research Center, linking patient records from two independent sources using probabilistic linkage as a baseline. A transformer-based model, RoBERTa, was fine-tuned for blocking using sentence embeddings. For matching, several language models were experimented under fine-tuned and zero-shot settings, assessing their performance against ground truth labels. Results: The fine-tuned blocking model achieved a 92% reduction in the number of candidate pairs while maintaining near-perfect recall. In the matching task, fine-tuned Mistral-7B achieved the best performance with only 6 incorrect predictions. Among zero-shot models, Mistral-Small-24B performed best, with a total of 55 incorrect predictions. Discussion: Fine-tuned language models achieved strong performance in patient record blocking and matching with minimal errors. However, they remain less accurate and efficient than a hybrid rule-based and probabilistic approach for blocking. Additionally, reasoning models like DeepSeek-R1 are impractical for large-scale record linkage due to high computational costs. Conclusion: This study highlights the potential of language models for automating patient record linkage, offering improved efficiency by eliminating the manual efforts required to perform patient record linkage. Overall, language models offer a scalable solution that can enhance data integration, reduce manual effort, and support disease surveillance and research. 

**Abstract (ZH)**: 客观目标：医疗数据碎片化为连接患者数据带来了重大挑战，需要强大的记录链接技术来整合来自多种来源的患者记录。本研究探讨了利用语言模型进行自动化患者记录链接的可行性，重点在于两个关键任务：阻止和匹配。材料与方法：我们使用了来自密苏里癌症注册与研究中心的真实医疗数据，使用概率链接作为基线，将患者的记录从两个独立来源链接起来。为了阻止阶段，我们使用了基于句子嵌入的RoBERTa模型进行了微调。对于匹配阶段，在微调和零样本设置下试验了多种语言模型，并评估了它们与真实标签的性能。结果：微调后的阻止模型在保持近乎完美的召回率的同时，将候选对的数量减少了92%。在匹配任务中，微调后的Mistral-7B表现出最佳性能，仅有6个错误预测。在零样本模型中，Mistral-Small-24B表现最佳，总共55个错误预测。讨论：微调后的语言模型在患者记录阻止和匹配任务中表现出较强的性能，并且错误较少。然而，它们在阻止阶段的准确性和效率仍然不如基于规则和概率的混合方法。另外，由于高计算成本，像DeepSeek-R1这样的推理模型对于大规模记录链接来说不切实际。结论：本研究强调了语言模型在自动化患者记录链接方面的潜力，通过消除手工进行患者记录链接所需的努力，提高了效率。总体来说，语言模型提供了一个可扩展的解决方案，能够增强数据整合，减少手工努力，并支持疾病监测和研究。 

---
# FlowReasoner: Reinforcing Query-Level Meta-Agents 

**Title (ZH)**: FlowReasoner: 强化查询级别元代理 

**Authors**: Hongcheng Gao, Yue Liu, Yufei He, Longxu Dou, Chao Du, Zhijie Deng, Bryan Hooi, Min Lin, Tianyu Pang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15257)  

**Abstract**: This paper proposes a query-level meta-agent named FlowReasoner to automate the design of query-level multi-agent systems, i.e., one system per user query. Our core idea is to incentivize a reasoning-based meta-agent via external execution feedback. Concretely, by distilling DeepSeek R1, we first endow the basic reasoning ability regarding the generation of multi-agent systems to FlowReasoner. Then, we further enhance it via reinforcement learning (RL) with external execution feedback. A multi-purpose reward is designed to guide the RL training from aspects of performance, complexity, and efficiency. In this manner, FlowReasoner is enabled to generate a personalized multi-agent system for each user query via deliberative reasoning. Experiments on both engineering and competition code benchmarks demonstrate the superiority of FlowReasoner. Remarkably, it surpasses o1-mini by 10.52% accuracy across three benchmarks. The code is available at this https URL. 

**Abstract (ZH)**: 本文提出了一种查询级元代理FlowReasoner，用于自动化查询级多代理系统的设计，即每个用户查询一个系统。我们的核心思想是通过外部执行反馈激励基于推理的元代理。具体而言，通过精炼DeepSeek R1，我们首先为FlowReasoner赋予了生成多代理系统的基本推理能力，然后进一步通过强化学习（RL）和外部执行反馈对其进行增强。设计了一个多用途奖励来从性能、复杂性和效率方面指导RL训练。通过这种方式，FlowReasoner能够通过审慎推理为每个用户查询生成个性化多代理系统。在工程和竞赛代码基准上的实验展示了FlowReasoner的优势。值得注意的是，它在三个基准上的准确率比o1-mini高10.52%。代码可在以下链接获取。 

---
# SuoiAI: Building a Dataset for Aquatic Invertebrates in Vietnam 

**Title (ZH)**: SuoiAI: 建立越南水生无脊椎动物数据集 

**Authors**: Tue Vo, Lakshay Sharma, Tuan Dinh, Khuong Dinh, Trang Nguyen, Trung Phan, Minh Do, Duong Vu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15252)  

**Abstract**: Understanding and monitoring aquatic biodiversity is critical for ecological health and conservation efforts. This paper proposes SuoiAI, an end-to-end pipeline for building a dataset of aquatic invertebrates in Vietnam and employing machine learning (ML) techniques for species classification. We outline the methods for data collection, annotation, and model training, focusing on reducing annotation effort through semi-supervised learning and leveraging state-of-the-art object detection and classification models. Our approach aims to overcome challenges such as data scarcity, fine-grained classification, and deployment in diverse environmental conditions. 

**Abstract (ZH)**: 理解与监测水生生物多样性对于生态健康和保护工作至关重要。本文提出了一种端到端的管道SuoiAI，用于构建越南水生无脊椎动物的数据集，并采用机器学习技术进行物种分类。我们概述了数据收集、标注和模型训练的方法，重点关注通过半监督学习减少标注努力，并利用最先进的对象检测和分类模型。我们的方法旨在克服数据稀缺、精细分类以及在多种环境条件下部署的挑战。 

---
# A Self-Improving Coding Agent 

**Title (ZH)**: 自我提升编码代理 

**Authors**: Maxime Robeyns, Martin Szummer, Laurence Aitchison  

**Link**: [PDF](https://arxiv.org/pdf/2504.15228)  

**Abstract**: We demonstrate that an LLM coding agent, equipped with basic coding tools, can autonomously edit itself, and thereby improve its performance on benchmark tasks. We find performance gains from 17% to 53% on a random subset of SWE Bench Verified, with additional performance gains on LiveCodeBench, as well as synthetically generated agent benchmarks. Our work represents an advancement in the automated and open-ended design of agentic systems, and provides a reference agent framework for those seeking to post-train LLMs on tool use and other agentic tasks. 

**Abstract (ZH)**: 我们展示了配备基本编码工具的LLM编码代理能够自主编辑自身，并因此在基准任务上改进其性能。我们在SWE Bench Verified的随机子集中获得了17%至53%的性能提升，在LiveCodeBench以及合成生成的代理基准上也获得了额外的性能提升。我们的工作代表了自动化和开放性设计代理系统的一个进步，并为那些寻求在工具使用和其他代理任务上后训练LLM的人提供了参考代理框架。 

---
# Position: Bayesian Statistics Facilitates Stakeholder Participation in Evaluation of Generative AI 

**Title (ZH)**: 位置：贝叶斯统计促进生成式人工智能评价中的利益相关者参与 

**Authors**: Yanan Long  

**Link**: [PDF](https://arxiv.org/pdf/2504.15211)  

**Abstract**: The evaluation of Generative AI (GenAI) systems plays a critical role in public policy and decision-making, yet existing methods are often limited by reliance on benchmark-driven, point-estimate comparisons that fail to capture uncertainty and broader societal impacts. This paper argues for the use of Bayesian statistics as a principled framework to address these challenges. Bayesian methods enable the integration of domain expertise through prior elicitation, allow for continuous learning from new data, and provide robust uncertainty quantification via posterior inference. We demonstrate how Bayesian inference can be applied to GenAI evaluation, particularly in incorporating stakeholder perspectives to enhance fairness, transparency, and reliability. Furthermore, we discuss Bayesian workflows as an iterative process for model validation and refinement, ensuring robust assessments of GenAI systems in dynamic, real-world contexts. 

**Abstract (ZH)**: 基于贝叶斯统计的方法在评估生成式人工智能系统中的应用：克服现有方法的局限以应对公共政策和决策中的挑战 

---
# Synergistic Weak-Strong Collaboration by Aligning Preferences 

**Title (ZH)**: 协同偏好对齐的弱强协作 

**Authors**: Yizhu Jiao, Xuchao Zhang, Zhaoyang Wang, Yubo Ma, Zhun Deng, Rujia Wang, Chetan Bansal, Saravan Rajmohan, Jiawei Han, Huaxiu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2504.15188)  

**Abstract**: Current Large Language Models (LLMs) excel in general reasoning yet struggle with specialized tasks requiring proprietary or domain-specific knowledge. Fine-tuning large models for every niche application is often infeasible due to black-box constraints and high computational overhead. To address this, we propose a collaborative framework that pairs a specialized weak model with a general strong model. The weak model, tailored to specific domains, produces initial drafts and background information, while the strong model leverages its advanced reasoning to refine these drafts, extending LLMs' capabilities to critical yet specialized tasks. To optimize this collaboration, we introduce a collaborative feedback to fine-tunes the weak model, which quantifies the influence of the weak model's contributions in the collaboration procedure and establishes preference pairs to guide preference tuning of the weak model. We validate our framework through experiments on three domains. We find that the collaboration significantly outperforms each model alone by leveraging complementary strengths. Moreover, aligning the weak model with the collaborative preference further enhances overall performance. 

**Abstract (ZH)**: 当前大型语言模型在通用推理方面表现出色，但在需要专有或领域特定知识的专业任务中却束手无策。由于黑箱约束和高计算开销，为每一个专门应用Fine-tune大模型往往不可行。为了解决这一问题，我们提出了一种协作框架，将专门的弱模型与通用的强模型配对。弱模型针对特定领域定制，生成初步草稿和背景信息，而强模型利用其先进的推理能力对这些草稿进行精炼，从而扩展大语言模型在关键且专门任务上的能力。为了优化这种协作，我们引入了一种协作反馈机制来Fine-tune弱模型，该机制量化弱模型贡献在协作过程中的影响，并建立偏好对来指导弱模型的偏好调优。我们通过在三个领域进行的实验验证了该框架。结果显示，协作显著优于单一模型，通过利用各自的互补优势。此外，使弱模型与协作偏好保持一致进一步提高了整体性能。 

---
# Behavioral Universe Network (BUN): A Behavioral Information-Based Framework for Complex Systems 

**Title (ZH)**: 行为宇宙网络(BUN):一种基于行为信息的复杂系统框架 

**Authors**: Wei Zhou, Ailiya Borjigin, Cong He  

**Link**: [PDF](https://arxiv.org/pdf/2504.15146)  

**Abstract**: Modern digital ecosystems feature complex, dynamic interactions among autonomous entities across diverse domains. Traditional models often separate agents and objects, lacking a unified foundation to capture their interactive behaviors. This paper introduces the Behavioral Universe Network (BUN), a theoretical framework grounded in the Agent-Interaction-Behavior (AIB) formalism. BUN treats subjects (active agents), objects (resources), and behaviors (operations) as first-class entities, all governed by a shared Behavioral Information Base (BIB). We detail the AIB core concepts and demonstrate how BUN leverages information-driven triggers, semantic enrichment, and adaptive rules to coordinate multi-agent systems. We highlight key benefits: enhanced behavior analysis, strong adaptability, and cross-domain interoperability. We conclude by positioning BUN as a promising foundation for next-generation digital governance and intelligent applications. 

**Abstract (ZH)**: 现代数字生态系统特征在于跨多个领域中自主实体之间的复杂动态交互。传统模型常常将代理和对象分开，缺乏一个统一的基础来捕捉它们的交互行为。本文介绍了一种基于代理-交互-行为（AIB）形式主义的理论框架——行为宇宙网络（BUN）。BUN将主体（活跃的代理）、对象（资源）和行为（操作）视为一级实体，并均由共享的行为信息库（BIB）管理。我们详细阐述了AIB的核心概念，并展示了BUN如何利用信息驱动的触发机制、语义增强和适应性规则来协调多代理系统。我们强调了BUN的关键优势：增强的行为分析能力、强大的适应性和跨领域的互操作性。最后，我们将BUN定位为下一代数字治理和智能应用的有前途的基础。 

---
# Contemplative Wisdom for Superalignment 

**Title (ZH)**: 观照智慧促进超对齐 

**Authors**: Ruben Laukkonen, Fionn Inglis, Shamil Chandaria, Lars Sandved-Smith, Jakob Hohwy, Jonathan Gold, Adam Elwood  

**Link**: [PDF](https://arxiv.org/pdf/2504.15125)  

**Abstract**: As artificial intelligence (AI) improves, traditional alignment strategies may falter in the face of unpredictable self-improvement, hidden subgoals, and the sheer complexity of intelligent systems. Rather than externally constraining behavior, we advocate designing AI with intrinsic morality built into its cognitive architecture and world model. Inspired by contemplative wisdom traditions, we show how four axiomatic principles can instil a resilient Wise World Model in AI systems. First, mindfulness enables self-monitoring and recalibration of emergent subgoals. Second, emptiness forestalls dogmatic goal fixation and relaxes rigid priors. Third, non-duality dissolves adversarial self-other boundaries. Fourth, boundless care motivates the universal reduction of suffering. We find that prompting AI to reflect on these principles improves performance on the AILuminate Benchmark using GPT-4o, particularly when combined. We offer detailed implementation strategies for state-of-the-art models, including contemplative architectures, constitutions, and reinforcement of chain-of-thought. For future systems, the active inference framework may offer the self-organizing and dynamic coupling capabilities needed to enact these insights in embodied agents. This interdisciplinary approach offers a self-correcting and resilient alternative to prevailing brittle control schemes. 

**Abstract (ZH)**: 随着人工智能（AI）的进步，传统的对齐策略可能在面对不可预测的自我提升、隐藏的子目标以及智能系统本身的复杂性时失效。我们主张设计具有内在道德观的AI，将其融入认知架构和世界模型中。受沉思智慧传统启发，我们展示了四种公理原则如何 instil 赋予AI系统一个有韧性的明智世界模型。首先，正念使自我监控和调整新兴的子目标成为可能。第二，空性防止固执的目标固定并放松坚定的先验假设。第三，不二消解了对抗性的自我与他者边界。第四，无限的关怀激励普遍减少苦痛。我们发现，促使AI反思这些原则可以提升使用GPT-4o在AILuminate基准测试中的性能，尤其是结合使用时。我们提供了对最新模型的详细实现策略，包括沉思架构、宪法以及对思维链的强化。对于未来的系统，主动推理框架可能提供自我组织和动态耦合的能力，以在实体代理中实现这些见解。这种跨学科的方法提供了一种自我校正和有韧性的替代方案，而不是当前脆弱的控制方案。 

---
# Mitigating Degree Bias in Graph Representation Learning with Learnable Structural Augmentation and Structural Self-Attention 

**Title (ZH)**: 使用可学习的结构增强和结构自注意力减轻图表示学习中的度偏差 

**Authors**: Van Thuy Hoang, Hyeon-Ju Jeon, O-Joun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.15075)  

**Abstract**: Graph Neural Networks (GNNs) update node representations through message passing, which is primarily based on the homophily principle, assuming that adjacent nodes share similar features. However, in real-world graphs with long-tailed degree distributions, high-degree nodes dominate message passing, causing a degree bias where low-degree nodes remain under-represented due to inadequate messages. The main challenge in addressing degree bias is how to discover non-adjacent nodes to provide additional messages to low-degree nodes while reducing excessive messages for high-degree nodes. Nevertheless, exploiting non-adjacent nodes to provide valuable messages is challenging, as it could generate noisy information and disrupt the original graph structures. To solve it, we propose a novel Degree Fairness Graph Transformer, named DegFairGT, to mitigate degree bias by discovering structural similarities between non-adjacent nodes through learnable structural augmentation and structural self-attention. Our key idea is to exploit non-adjacent nodes with similar roles in the same community to generate informative edges under our augmentation, which could provide informative messages between nodes with similar roles while ensuring that the homophily principle is maintained within the community. To enable DegFairGT to learn such structural similarities, we then propose a structural self-attention to capture the similarities between node pairs. To preserve global graph structures and prevent graph augmentation from hindering graph structure, we propose a Self-Supervised Learning task to preserve p-step transition probability and regularize graph augmentation. Extensive experiments on six datasets showed that DegFairGT outperformed state-of-the-art baselines in degree fairness analysis, node classification, and node clustering tasks. 

**Abstract (ZH)**: Degree公平图变换器（DegFairGT）：通过发现非相邻节点的结构性相似性来减轻度偏差 

---
# Text-to-Decision Agent: Learning Generalist Policies from Natural Language Supervision 

**Title (ZH)**: 文本决策代理：从自然语言监督学习通用策略 

**Authors**: Shilin Zhang, Zican Hu, Wenhao Wu, Xinyi Xie, Jianxiang Tang, Chunlin Chen, Daoyi Dong, Yu Cheng, Zhenhong Sun, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15046)  

**Abstract**: RL systems usually tackle generalization by inferring task beliefs from high-quality samples or warmup explorations. The restricted form limits their generality and usability since these supervision signals are expensive and even infeasible to acquire in advance for unseen tasks. Learning directly from the raw text about decision tasks is a promising alternative to leverage a much broader source of supervision. In the paper, we propose Text-to-Decision Agent (T2DA), a simple and scalable framework that supervises generalist policy learning with natural language. We first introduce a generalized world model to encode multi-task decision data into a dynamics-aware embedding space. Then, inspired by CLIP, we predict which textual description goes with which decision embedding, effectively bridging their semantic gap via contrastive language-decision pre-training and aligning the text embeddings to comprehend the environment dynamics. After training the text-conditioned generalist policy, the agent can directly realize zero-shot text-to-decision generation in response to language instructions. Comprehensive experiments on MuJoCo and Meta-World benchmarks show that T2DA facilitates high-capacity zero-shot generalization and outperforms various types of baselines. 

**Abstract (ZH)**: RL系统通常通过从高质量样本或暖启动探索中推断任务信念来应对泛化问题。这种限制性形式限制了其通用性和实用性，因为这些监督信号对于未见过的任务来说在事前获取往往是昂贵的甚至不可行的。直接从原始文本中学习决策任务是一种有希望的替代方案，可以利用更广泛来源的监督。在本文中，我们提出了Text-to-Decision Agent (T2DA)，这是一种简单且可扩展的框架，利用自然语言监督通用策略的學習。我们首先介绍了一个通用的世界模型，将多任务决策数据编码到动力感知的嵌入空间中。然后，受到CLIP的启发，我们预测哪些文本描述与哪个决策嵌入相关联，通过对比语言-决策预训练有效地弥合了它们之间的语义差距，并使文本嵌入能够理解环境动力学。在训练文本条件下的通用策略后，代理可以直接在收到语言指令时实现零样本的文本到决策生成。在MuJoCo和Meta-World基准上的全面实验表明，T2DA 支持高度容量的零样本泛化，并优于各种基线。 

---
# Evaluating Code Generation of LLMs in Advanced Computer Science Problems 

**Title (ZH)**: 评估先进计算机科学问题中LLM代码生成能力 

**Authors**: Emir Catir, Robin Claesson, Rodothea Myrsini Tsoupidi  

**Link**: [PDF](https://arxiv.org/pdf/2504.14964)  

**Abstract**: Large Language Models (LLMs), such as GitHub Copilot and ChatGPT have become popular among programming students. Students use LLMs to assist them in programming courses, including generating source code. Previous work has evaluated the ability of LLMs in solving introductory-course programming assignments. The results have shown that LLMs are highly effective in generating code for introductory Computer Science (CS) courses. However, there is a gap in research on evaluating LLMs' ability to generate code that solves advanced programming assignments. In this work, we evaluate the ability of four LLM tools to solve programming assignments from advanced CS courses in three popular programming languages, Java, Python, and C. We manually select 12 problems, three problems from introductory courses as the baseline and nine programming assignments from second- and third-year CS courses. To evaluate the LLM-generated code, we generate a test suite of 1000 test cases per problem and analyze the program output. Our evaluation shows that although LLMs are highly effective in generating source code for introductory programming courses, solving advanced programming assignments is more challenging. Nonetheless, in many cases, LLMs identify the base problem and provide partial solutions that may be useful to CS students. Furthermore, our results may provide useful guidance for teachers of advanced programming courses on how to design programming assignments. 

**Abstract (ZH)**: 大型语言模型（LLMs）如GitHub Copilot和ChatGPT在编程学生中日益流行。学生使用LLMs辅助他们在编程课程中生成源代码。先前的研究评估了LLMs解决入门级编程作业的能力，结果显示，LLMs在生成计算科学（CS）入门课程的代码方面非常有效。然而，对评估LLMs生成解决高级编程作业代码能力的研究存在空白。在这项工作中，我们评估了四个LLM工具解决使用Java、Python和C三种流行编程语言的高级CS课程编程作业的能力。我们手动选择了12个问题，包括三个入门课程的问题作为基线，以及九个来自二年级和三年级CS课程的编程作业。为了评估LLM生成的代码，我们为每个问题生成了1000个测试用例，并分析程序输出。我们的评估显示，尽管LLMs在生成入门级编程课程的源代码方面非常有效，但解决高级编程作业更具挑战性。然而，在许多情况下，LLMs能够识别基础问题并提供可能对计算机科学学生有用的部分解决方案。此外，我们的结果可能为高级编程课程教师设计编程作业提供有价值的指导。 

---
# Generative Semantic Communications: Principles and Practices 

**Title (ZH)**: 生成性语义通信：原理与实践 

**Authors**: Xiaojun Yuan, Haoming Ma, Yinuo Huang, Zhoufan Hua, Yong Zuo, Zhi Ding  

**Link**: [PDF](https://arxiv.org/pdf/2504.14947)  

**Abstract**: Semantic communication leverages artificial intelligence (AI) technologies to extract semantic information from data for efficient transmission, theraby significantly reducing communication cost. With the evolution towards artificial general intelligence (AGI), the increasing demands for AGI services pose new challenges to semantic communication. In response, we propose a new paradigm for AGI-driven communications, called generative semantic communication (GSC), which utilizes advanced AI technologies such as foundation models and generative models. We first describe the basic concept of GSC and its difference from existing semantic communications, and then introduce a general framework of GSC, followed by two case studies to verify the advantages of GSC in AGI-driven applications. Finally, open challenges and new research directions are discussed to stimulate this line of research and pave the way for practical applications. 

**Abstract (ZH)**: 基于人工智能的语义通信利用人工智能技术从数据中提取语义信息以实现高效传输，从而显著降低通信成本。随着通向通用人工智能（AGI）的演进，AGI服务的需求增长为语义通信带来了新的挑战。为此，我们提出了一种新的AGI驱动通信范式，称为生成性语义通信（GSC），并利用诸如基础模型和生成模型等先进人工智能技术。首先描述了GSC的基本概念及其与现有语义通信的区别，然后介绍了一般框架，并通过两个案例研究验证了GSC在AGI驱动应用中的优势。最后，讨论了开放性挑战和新的研究方向，以促进这一研究线的发展并为实际应用铺平道路。 

---
# EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework 

**Title (ZH)**: 教育Q：通过多代理对话框架评估LLM的教学能力 

**Authors**: Yao Shi, Rongkeng Liang, Yong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14928)  

**Abstract**: Large language models (LLMs) increasingly serve as educational tools, yet evaluating their teaching capabilities remains challenging due to the resource-intensive, context-dependent, and methodologically complex nature of teacher-student interactions. We introduce EducationQ, a multi-agent dialogue framework that efficiently assesses teaching capabilities through simulated dynamic educational scenarios, featuring specialized agents for teaching, learning, and evaluation. Testing 14 LLMs across major AI Organizations (OpenAI, Meta, Google, Anthropic, and others) on 1,498 questions spanning 13 disciplines and 10 difficulty levels reveals that teaching effectiveness does not correlate linearly with model scale or general reasoning capabilities - with some smaller open-source models outperforming larger commercial counterparts in teaching contexts. This finding highlights a critical gap in current evaluations that prioritize knowledge recall over interactive pedagogy. Our mixed-methods evaluation, combining quantitative metrics with qualitative analysis and expert case studies, identifies distinct pedagogical strengths employed by top-performing models (e.g., sophisticated questioning strategies, adaptive feedback mechanisms). Human expert evaluations show 78% agreement with our automated qualitative analysis of effective teaching behaviors, validating our methodology. EducationQ demonstrates that LLMs-as-teachers require specialized optimization beyond simple scaling, suggesting next-generation educational AI prioritize targeted enhancement of specific pedagogical effectiveness. 

**Abstract (ZH)**: 大型语言模型（LLMs）在教育领域日益发挥重要作用，但由于教师-学生互动的资源密集、情境依赖和方法复杂性，评估其教学能力仍具挑战性。我们引入了EducationQ，一种通过模拟动态教育场景来高效评估教学能力的多智能体对话框架，包含专门的教学、学习和评价代理。对来自主要AI组织（OpenAI、Meta、Google、Anthropic及其他）的14个LLM在涵盖13个学科和10个难度等级的1,498个问题上的测试结果显示，教学效果与模型规模或一般推理能力之间并不存在线性关系——一些较小的开源模型在教学场景中的表现优于大型商业模型。这一发现突显了当前评估中的关键缺口，即过于强调知识回忆而忽视了互动式教学方法。我们的混合方法评估结合定量指标、定性分析和专家案例研究，识别出顶级模型在特定教学方法上的独特优势（如复杂的提问策略、适应性反馈机制）。人类专家评估结果显示，78%的人赞同我们对有效教学行为的自动化定性分析，验证了我们的方法论。EducationQ表明，作为教师的LLMs需要超出简单扩增的专业优化，建议下一代教育AI重点关注特定教学效果的针对性提升。 

---
# OTC: Optimal Tool Calls via Reinforcement Learning 

**Title (ZH)**: OTC：通过强化学习实现最优工具调用 

**Authors**: Hongru Wang, Cheng Qian, Wanjun Zhong, Xiusi Chen, Jiahao Qiu, Shijue Huang, Bowen Jin, Mengdi Wang, Kam-Fai Wong, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.14870)  

**Abstract**: Tool-integrated reasoning (TIR) augments large language models (LLMs) with the ability to invoke external tools, such as search engines and code interpreters, to solve tasks beyond the capabilities of language-only reasoning. While reinforcement learning (RL) has shown promise in improving TIR by optimizing final answer correctness, existing approaches often overlook the efficiency and cost associated with tool usage. This can lead to suboptimal behavior, including excessive tool calls that increase computational and financial overhead, or insufficient tool use that compromises answer quality. In this work, we propose Optimal Tool Call-controlled Policy Optimization (OTC-PO), a simple yet effective RL-based framework that encourages models to produce accurate answers with minimal tool calls. Our method introduces a tool-integrated reward that jointly considers correctness and tool efficiency, promoting high tool productivity. We instantiate this framework within both Proximal Policy Optimization (PPO) and Group Relative Preference Optimization (GRPO), resulting in OTC-PPO and OTC-GRPO. Experiments with Qwen-2.5 and Qwen-Math across multiple QA benchmarks show that our approach reduces tool calls by up to 73.1\% and improves tool productivity by up to 229.4\%, while maintaining comparable answer accuracy. To the best of our knowledge, this is the first RL-based framework that explicitly optimizes tool-use efficiency in TIR. 

**Abstract (ZH)**: 基于工具优化的强化学习策略优化（OTC-PO）：一种简单有效的工具集成推理框架 

---
# AlignRAG: An Adaptable Framework for Resolving Misalignments in Retrieval-Aware Reasoning of RAG 

**Title (ZH)**: AlignRAG: 一种用于解决RAG检索aware推理中不一致性的可适应框架 

**Authors**: Jiaqi Wei, Hao Zhou, Xiang Zhang, Di Zhang, Zijie Qiu, Wei Wei, Jinzhe Li, Wanli Ouyang, Siqi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.14858)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as a foundational paradigm for knowledge-grounded text generation. However, existing RAG pipelines often fail to ensure that the reasoning trajectories align with the evidential constraints imposed by retrieved content. In this paper, we reframe RAG as a problem of retrieval-aware reasoning and identify a core challenge: reasoning misalignment-the mismatch between a model's reasoning trajectory and the retrieved evidence. To address this challenge, we propose AlignRAG, a novel test-time framework that mitigates reasoning misalignment through iterative Critique-Driven Alignment (CDA) steps. In contrast to prior approaches that rely on static training or post-hoc selection, AlignRAG actively refines reasoning trajectories during inference by enforcing fine-grained alignment with evidence. Our framework introduces a new paradigm for retrieval-aware reasoning by: (1) constructing context-rich training corpora; (2) generating contrastive critiques from preference-aware reasoning trajectories; (3) training a dedicated \textit{Critic Language Model (CLM)} to identify reasoning misalignments; and (4) applying CDA steps to optimize reasoning trajectories iteratively. Empirical results demonstrate that AlignRAG consistently outperforms all baselines and could integrate as a plug-and-play module into existing RAG pipelines without further changes. By reconceptualizing RAG as a structured reasoning trajectory and establishing the test-time framework for correcting reasoning misalignments in RAG, AlignRAG provides practical advancements for retrieval-aware generation. 

**Abstract (ZH)**: 检索增强生成（RAG）已成为基于知识的文本生成的基础范式。然而，现有的RAG流水线往往无法确保推理轨迹与检索内容施加的证据约束保持一致。在这个论文中，我们将RAG重新框架化为一种检索意识推理问题，并识别出一个核心挑战：推理错位——模型的推理轨迹与检索证据之间的不匹配。为了解决这一挑战，我们提出了AlignRAG，这是一种新颖的测试时框架，通过迭代的质疑驱动校准（CDA）步骤来缓解推理错位。与依赖于静态训练或事后选择的先前方法不同，AlignRAG在推断过程中主动通过细粒度证据校准来精炼推理轨迹。我们的框架通过以下方式引入了检索意识推理的新范式：（1）构建丰富的上下文训练语料库；（2）从偏好意识推理轨迹中生成对比性批评；（3）训练专用的批评语言模型（CLM）以识别推理错位；（4）应用CDA步骤以迭代优化推理轨迹。实验证明，AlignRAG始终优于所有基线，在无需进一步修改的情况下可以无缝集成到现有的RAG流水线中。通过将RAG重新构想为有结构的推理轨迹，并建立纠正RAG中推理错位的测试时框架，AlignRAG为检索意识生成提供了实用的进步。 

---
# Establishing Reliability Metrics for Reward Models in Large Language Models 

**Title (ZH)**: 大型语言模型中奖励模型可靠性的评价指标建立 

**Authors**: Yizhou Chen, Yawen Liu, Xuesi Wang, Qingtao Yu, Guangda Huzhang, Anxiang Zeng, Han Yu, Zhiming Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.14838)  

**Abstract**: The reward model (RM) that represents human preferences plays a crucial role in optimizing the outputs of large language models (LLMs), e.g., through reinforcement learning from human feedback (RLHF) or rejection sampling. However, a long challenge for RM is its uncertain reliability, i.e., LLM outputs with higher rewards may not align with actual human preferences. Currently, there is a lack of a convincing metric to quantify the reliability of RMs. To bridge this gap, we propose the \textit{\underline{R}eliable at \underline{$\eta$}} (RETA) metric, which directly measures the reliability of an RM by evaluating the average quality (scored by an oracle) of the top $\eta$ quantile responses assessed by an RM. On top of RETA, we present an integrated benchmarking pipeline that allows anyone to evaluate their own RM without incurring additional Oracle labeling costs. Extensive experimental studies demonstrate the superior stability of RETA metric, providing solid evaluations of the reliability of various publicly available and proprietary RMs. When dealing with an unreliable RM, we can use the RETA metric to identify the optimal quantile from which to select the responses. 

**Abstract (ZH)**: 可靠的η分位数响应评价(RETA)指标 

---
# DONOD: Robust and Generalizable Instruction Fine-Tuning for LLMs via Model-Intrinsic Dataset Pruning 

**Title (ZH)**: DONOD：通过模型固有数据集修剪实现的LLMs鲁棒且可泛化的指令微调 

**Authors**: Jucheng Hu, Surong Yang, Dongzhan Zhou, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14810)  

**Abstract**: Ad-hoc instruction fine-tuning of large language models (LLMs) is widely adopted for domain-specific adaptation. While domain-specific supervised fine-tuning (SFT) is effective and efficient, it often weakens cross-domain generalization and struggles with noisy training data. To address these challenges, we propose DONOD, a lightweight model-intrinsic data pruning method. Our approach evaluates data using two model-parameter-based metrics: Delta of Norm (DON), which captures the cumulative influence on model weights, and Norm of Delta (NOD), which quantifies weight instability. Moreover, by employing the Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS) algorithm, we effectively filter noisy, unlearnable, and generalization-harming samples without relying on auxiliary models during the SFT process. Experiments on mathematical tasks demonstrate that data selected by DONOD achieve superior fine-tuning efficiency and improved robustness against noisy data. By filtering out 70% of the full dataset, we improve target-domain accuracy by 14.90% and cross-domain accuracy by 5.67%. Meanwhile, our selected data present superior cross-architecture generalization. Data pruned by smaller models (e.g., Llama 3.1-8B) generalize effectively on larger models (e.g., Llama 2-13B). Compared to existing related methodologies, DONOD demonstrates comparable or superior performance while remaining dataset-agnostic, enabling broader applicability. 

**Abstract (ZH)**: 自适应指令微调大语言模型的小型化模型内在数据剪枝方法 

---
# PLANET: A Collection of Benchmarks for Evaluating LLMs' Planning Capabilities 

**Title (ZH)**: PLANET: 评估大规模语言模型规划能力的标准集合 

**Authors**: Haoming Li, Zhaoliang Chen, Jonathan Zhang, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14773)  

**Abstract**: Planning is central to agents and agentic AI. The ability to plan, e.g., creating travel itineraries within a budget, holds immense potential in both scientific and commercial contexts. Moreover, optimal plans tend to require fewer resources compared to ad-hoc methods. To date, a comprehensive understanding of existing planning benchmarks appears to be lacking. Without it, comparing planning algorithms' performance across domains or selecting suitable algorithms for new scenarios remains challenging. In this paper, we examine a range of planning benchmarks to identify commonly used testbeds for algorithm development and highlight potential gaps. These benchmarks are categorized into embodied environments, web navigation, scheduling, games and puzzles, and everyday task automation. Our study recommends the most appropriate benchmarks for various algorithms and offers insights to guide future benchmark development. 

**Abstract (ZH)**: 规划对于代理和代理人工智能是至关重要的。规划能力，例如在预算内的行程规划，具有在科学和商业领域巨大的潜力。此外，最优规划通常会比即兴方法更节约资源。迄今为止，对现有规划基准的全面理解似乎仍然不足。缺乏这种理解，跨领域比较规划算法的性能或为新场景选择合适的算法仍然具有挑战性。在本文中，我们研究了一系列规划基准，以识别算法开发中常用的标准测试平台，并指出潜在的差距。这些基准划分为实体环境、网络导航、调度、游戏和谜题以及日常任务自动化。我们的研究建议适合各种算法的最佳基准，并提供指导未来基准开发的见解。 

---
# AI with Emotions: Exploring Emotional Expressions in Large Language Models 

**Title (ZH)**: 具有情感的AI：探索大规模语言模型中的情感表达 

**Authors**: Shin-nosuke Ishikawa, Atsushi Yoshino  

**Link**: [PDF](https://arxiv.org/pdf/2504.14706)  

**Abstract**: The human-level performance of Large Language Models (LLMs) across various tasks has raised expectations for the potential of Artificial Intelligence (AI) to possess emotions someday. To explore the capability of current LLMs to express emotions in their outputs, we conducted an experiment using several LLMs (OpenAI GPT, Google Gemini, Meta Llama3, and Cohere Command R+) to role-play as agents answering questions with specified emotional this http URL defined the emotional states using Russell's Circumplex model, a well-established framework that characterizes emotions along the sleepy-activated (arousal) and pleasure-displeasure (valence) axes. We chose this model for its simplicity, utilizing two continuous parameters, which allows for better controllability in applications involving continuous changes in emotional states. The responses generated were evaluated using a sentiment analysis model, independent of the LLMs, trained on the GoEmotions dataset. The evaluation showed that the emotional states of the generated answers were consistent with the specifications, demonstrating the LLMs' capability for emotional expression. This indicates the potential for LLM-based AI agents to simulate emotions, opening up a wide range of applications for emotion-based interactions, such as advisors or consultants who can provide advice or opinions with a personal touch. 

**Abstract (ZH)**: 大型语言模型在各类任务中达到人类水平的表现引发了对未来人工智能具备情感可能性的期望。为了探究当前大型语言模型在输出中表达情感的能力，我们使用了几种大型语言模型（OpenAI GPT、Google Gemini、Meta Llama3 和 Cohere Command R+）进行角色扮演，使其以指定情感回答问题。我们使用拉塞尔环形模型定义情感状态，该模型是一个成熟的框架，沿唤醒-激活（唤醒度）和愉悦-不悦（价值度）两个轴来刻画情感。我们选择了该模型因为其简单性，使用了两个连续参数，这在涉及情感状态连续变化的应用中提供了更好的可控性。生成的响应使用与大型语言模型独立的基于情感分析的模型进行评估，该模型在GoEmotions数据集上进行了训练。评估结果显示生成的答案的情感状态与规定相符，证明了大型语言模型具有情感表达的能力。这表明基于大型语言模型的AI代理有模拟情感的潜力，为基于情感的交互提供了广泛的应用前景，如能够以个人化方式提供建议或意见的顾问或咨询师。 

---
# A Framework for Benchmarking and Aligning Task-Planning Safety in LLM-Based Embodied Agents 

**Title (ZH)**: 基于大语言模型的物理体代理任务规划安全性基准评估与对齐框架 

**Authors**: Yuting Huang, Leilei Ding, Zhipeng Tang, Tianfu Wang, Xinrui Lin, Wuyang Zhang, Mingxiao Ma, Yanyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14650)  

**Abstract**: Large Language Models (LLMs) exhibit substantial promise in enhancing task-planning capabilities within embodied agents due to their advanced reasoning and comprehension. However, the systemic safety of these agents remains an underexplored frontier. In this study, we present Safe-BeAl, an integrated framework for the measurement (SafePlan-Bench) and alignment (Safe-Align) of LLM-based embodied agents' behaviors. SafePlan-Bench establishes a comprehensive benchmark for evaluating task-planning safety, encompassing 2,027 daily tasks and corresponding environments distributed across 8 distinct hazard categories (e.g., Fire Hazard). Our empirical analysis reveals that even in the absence of adversarial inputs or malicious intent, LLM-based agents can exhibit unsafe behaviors. To mitigate these hazards, we propose Safe-Align, a method designed to integrate physical-world safety knowledge into LLM-based embodied agents while maintaining task-specific performance. Experiments across a variety of settings demonstrate that Safe-BeAl provides comprehensive safety validation, improving safety by 8.55 - 15.22%, compared to embodied agents based on GPT-4, while ensuring successful task completion. 

**Abstract (ZH)**: Safe-BeAl：一种LLM驱动的可信赖体化智能体的衡量与对齐框架 

---
# Consensus in Motion: A Case of Dynamic Rationality of Sequential Learning in Probability Aggregation 

**Title (ZH)**: 共识在motion：概率聚合中序贯学习动态理性案例研究 

**Authors**: Polina Gordienko, Christoph Jansen, Thomas Augustin, Martin Rechenauer  

**Link**: [PDF](https://arxiv.org/pdf/2504.14624)  

**Abstract**: We propose a framework for probability aggregation based on propositional probability logic. Unlike conventional judgment aggregation, which focuses on static rationality, our model addresses dynamic rationality by ensuring that collective beliefs update consistently with new information. We show that any consensus-compatible and independent aggregation rule on a non-nested agenda is necessarily linear. Furthermore, we provide sufficient conditions for a fair learning process, where individuals initially agree on a specified subset of propositions known as the common ground, and new information is restricted to this shared foundation. This guarantees that updating individual judgments via Bayesian conditioning-whether performed before or after aggregation-yields the same collective belief. A distinctive feature of our framework is its treatment of sequential decision-making, which allows new information to be incorporated progressively through multiple stages while maintaining the established common ground. We illustrate our findings with a running example in a political scenario concerning healthcare and immigration policies. 

**Abstract (ZH)**: 我们提出了一种基于命题概率逻辑的概率聚合框架。不同于侧重静态理性的传统判断聚合，我们的模型通过确保集体信念在获得新信息后能一致更新，来处理动态理性。我们证明，对于非嵌套议程上的任何共识兼容且独立的聚合规则，必定是线性的。此外，我们提供了确保公平学习过程的充分条件，在这种过程中，个体最初在一组特定命题上达成共识，这些命题被称为共同基础，而新的信息仅限于这一共享基础之上。这保证了通过贝叶斯条件化更新个体判断——无论是聚合前还是聚合后——都能得出相同的集体信念。我们的框架的一个独特之处在于其对顺序决策的处理，这使新的信息可以通过多个阶段逐步融入，同时保持已建立的共同基础。我们通过一个关于医疗政策和移民政策的政治场景实例来说明这些发现。 

---
# UFO2: The Desktop AgentOS 

**Title (ZH)**: UFO2：桌面代理操作系统 

**Authors**: Chaoyun Zhang, He Huang, Chiming Ni, Jian Mu, Si Qin, Shilin He, Lu Wang, Fangkai Yang, Pu Zhao, Chao Du, Liqun Li, Yu Kang, Zhao Jiang, Suzhen Zheng, Rujia Wang, Jiaxu Qian, Minghua Ma, Jian-Guang Lou, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14603)  

**Abstract**: Recent Computer-Using Agents (CUAs), powered by multimodal large language models (LLMs), offer a promising direction for automating complex desktop workflows through natural language. However, most existing CUAs remain conceptual prototypes, hindered by shallow OS integration, fragile screenshot-based interaction, and disruptive execution.
We present UFO2, a multiagent AgentOS for Windows desktops that elevates CUAs into practical, system-level automation. UFO2 features a centralized HostAgent for task decomposition and coordination, alongside a collection of application-specialized AppAgent equipped with native APIs, domain-specific knowledge, and a unified GUI--API action layer. This architecture enables robust task execution while preserving modularity and extensibility. A hybrid control detection pipeline fuses Windows UI Automation (UIA) with vision-based parsing to support diverse interface styles. Runtime efficiency is further enhanced through speculative multi-action planning, reducing per-step LLM overhead. Finally, a Picture-in-Picture (PiP) interface enables automation within an isolated virtual desktop, allowing agents and users to operate concurrently without interference.
We evaluate UFO2 across over 20 real-world Windows applications, demonstrating substantial improvements in robustness and execution accuracy over prior CUAs. Our results show that deep OS integration unlocks a scalable path toward reliable, user-aligned desktop automation. 

**Abstract (ZH)**: Recent Computer-Using Agents (CUAs) Powered by Multimodal Large Language Models for Robust Desktop Workflow Automation via Natural Language: The UFO2 Multiagent AgentOS for Windows Desktops 

---
# Toward the Axiomatization of Intelligence: Structure, Time, and Existence 

**Title (ZH)**: 智能的公理化：结构、时间与存在 

**Authors**: Kei Itoh  

**Link**: [PDF](https://arxiv.org/pdf/2504.14596)  

**Abstract**: This study aims to construct an axiomatic definition of intelligence within a meta-framework that defines the method of definition, addressing intelligence as an inherently naive and polysemous concept. Initially, we formalize a set-theoretic representation of the universe as the domain wherein intelligence exists and characterize intelligence as a structure that involves temporal evolution and interaction with other sets. Starting from a naive definition of intelligence as "an entity possessing structures for externally inputting, internally processing, and externally outputting information or matter," we axiomatically reformulate it within this set-theoretical depiction of the universe. Applying this axiomatic definition, we compare and interpret three examples -- Hebbian non-optimized neural networks (NNs), backpropagation-optimized NNs, and biological reflexive systems -- in terms of their intelligence, structural properties, and biological plausibility. Furthermore, by extending our definition into a categorical framework, we introduce two categories, "Time Category" and "Intelligence Category," along with the functorial relationships between them, demonstrating the potential to represent changes and mimicry relationships among intelligent systems abstractly. Additionally, since intelligence, as defined herein, functions effectively only when accompanied by temporal interactions, we introduce the concept of "activity" and explore how activity-based conditions influence classifications and interpretations of intelligence. Finally, we suggest that our definitional methodology is not limited to intelligence alone, but can be similarly applied to other concepts, such as consciousness and emotion, advocating for their formal reinterpretation through the same procedural steps: defining a universal representation, selecting naive definitions, and axiomatic formalization. 

**Abstract (ZH)**: 本研究旨在在元框架中构建智能的公理化定义，该框架定义了定义方法，以应对智能这一先天模糊和多义的概念。首先，我们形式化一个集合表示的宇宙，作为智能存在的领域，并将智能视为涉及时间演化和与其他集合交互的结构。从“智能是具有处理外部输入、内部处理和输出信息或物质的结构的实体”的朴素定义出发，我们在这种集合表示的宇宙中公理化地重新定义它。通过这种公理化定义，我们比较并解释三种示例——递推非优化神经网络（NNs）、反向传播优化NNs和生物反射系统——在智能、结构特性和生物可行性方面的差异。此外，通过将定义扩展到范畴框架，我们引入了两个范畴“时间范畴”和“智能范畴”，以及它们之间的函子关系，展示了如何抽象地表示智能系统的变化及其模仿关系。此外，由于在此定义下，智能的有效作用仅限于伴随时间交互时，我们引入了“活动”的概念，并探讨基于活动条件如何影响智能的分类和解释。最后，我们建议我们的定义方法不仅限于智能，还可以类似地应用于其他概念，如意识和情绪，倡导通过相同的程序步骤对其形式重解释：定义普遍表示、选择朴素定义和公理化形式化。 

---
# LLM-Enabled In-Context Learning for Data Collection Scheduling in UAV-assisted Sensor Networks 

**Title (ZH)**: 基于LLM的UAV辅助传感器网络中数据采集调度的上下文学习方法 

**Authors**: Yousef Emami, Hao Gao, SeyedSina Nabavirazani, Luis Almeida  

**Link**: [PDF](https://arxiv.org/pdf/2504.14556)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) are increasingly being used in various private and commercial applications, e.g. traffic control, package delivery, and Search and Rescue (SAR) operations. Machine Learning (ML) methods used in UAV-assisted Sensor Networks (UASNETs) and especially in Deep Reinforcement Learning (DRL) face challenges such as complex and lengthy model training, gaps between simulation and reality, and low sample efficiency, which conflict with the urgency of emergencies such as SAR operations. This paper proposes In-Context Learning (ICL)-based Data Collection Scheduling (ICLDC) scheme, as an alternative to DRL in emergencies. The UAV collects and transmits logged sensory data, to an LLM, to generate a task description in natural language, from which it obtains a data collection schedule to be executed by the UAV. The system continuously adapts by adding feedback to task descriptions and utilizing feedback for future decisions. This method is tested against jailbreaking attacks, where task description is manipulated to undermine network performance, highlighting the vulnerability of LLMs to such attacks. The proposed ICLDC outperforms the Maximum Channel Gain by reducing cumulative packet loss by approximately 56\%. ICLDC presents a promising direction for intelligent scheduling and control in UAV-assisted data collection. 

**Abstract (ZH)**: 无人航空 vehicles (UAVs) 在各种私人和商业应用中越来越广泛，例如交通控制、包裹配送和搜索与救援 (SAR) 操作中。UAV辅助传感器网络(UASNETs)和特别是在深度强化学习(DRL)中使用的机器学习(ML)方法面临着模型训练复杂且耗时、模拟与现实之间的差距以及样本效率低下的挑战，这些挑战与SAR等紧急操作的迫切性冲突。本文提出了一种基于上下文学习(In-Context Learning, ICL)-数据收集调度(ICL-DC)方案，作为一种紧急情况下的替代DRL方法。UAV收集并传输记录的传感数据至语言模型(LLM)，以生成自然语言的任务描述，从中获得由UAV执行的数据收集计划。系统通过不断添加反馈来调整任务描述，并利用反馈对未来决策进行优化。该方法针对越狱攻击进行了测试，在这种攻击中，通过操纵任务描述来削弱网络性能，突显了LLM对这类攻击的脆弱性。提出的ICL-DC通过将累积包丢失减少约56％，优于最大信道增益，展示了在UAV辅助数据收集中进行智能调度和控制的有前景的方向。 

---
# Learning from Reasoning Failures via Synthetic Data Generation 

**Title (ZH)**: 通过合成数据生成学习推理失败 

**Authors**: Gabriela Ben Melech Stan, Estelle Aflalo, Avinash Madasu, Vasudev Lal, Phillip Howard  

**Link**: [PDF](https://arxiv.org/pdf/2504.14523)  

**Abstract**: Training models on synthetic data has emerged as an increasingly important strategy for improving the performance of generative AI. This approach is particularly helpful for large multimodal models (LMMs) due to the relative scarcity of high-quality paired image-text data compared to language-only data. While a variety of methods have been proposed for generating large multimodal datasets, they do not tailor the synthetic data to address specific deficiencies in the reasoning abilities of LMMs which will be trained with the generated dataset. In contrast, humans often learn in a more efficient manner by seeking out examples related to the types of reasoning where they have failed previously. Inspired by this observation, we propose a new approach for synthetic data generation which is grounded in the analysis of an existing LMM's reasoning failures. Our methodology leverages frontier models to automatically analyze errors produced by a weaker LMM and propose new examples which can be used to correct the reasoning failure via additional training, which are then further filtered to ensure high quality. We generate a large multimodal instruction tuning dataset containing over 553k examples using our approach and conduct extensive experiments demonstrating its utility for improving the performance of LMMs on multiple downstream tasks. Our results show that models trained on our synthetic data can even exceed the performance of LMMs trained on an equivalent amount of additional real data, demonstrating the high value of generating synthetic data targeted to specific reasoning failure modes in LMMs. We will make our dataset and code publicly available. 

**Abstract (ZH)**: 基于现有大型多模态模型推理缺陷分析的合成数据生成方法 

---
# Meta-Thinking in LLMs via Multi-Agent Reinforcement Learning: A Survey 

**Title (ZH)**: 利用多代理强化学习实现LLMs的元思维：一个综述 

**Authors**: Ahsan Bilal, Muhammad Ahmed Mohsin, Muhammad Umer, Muhammad Awais Khan Bangash, Muhammad Ali Jamshed  

**Link**: [PDF](https://arxiv.org/pdf/2504.14520)  

**Abstract**: This survey explores the development of meta-thinking capabilities in Large Language Models (LLMs) from a Multi-Agent Reinforcement Learning (MARL) perspective. Meta-thinking self-reflection, assessment, and control of thinking processes is an important next step in enhancing LLM reliability, flexibility, and performance, particularly for complex or high-stakes tasks. The survey begins by analyzing current LLM limitations, such as hallucinations and the lack of internal self-assessment mechanisms. It then talks about newer methods, including RL from human feedback (RLHF), self-distillation, and chain-of-thought prompting, and each of their limitations. The crux of the survey is to talk about how multi-agent architectures, namely supervisor-agent hierarchies, agent debates, and theory of mind frameworks, can emulate human-like introspective behavior and enhance LLM robustness. By exploring reward mechanisms, self-play, and continuous learning methods in MARL, this survey gives a comprehensive roadmap to building introspective, adaptive, and trustworthy LLMs. Evaluation metrics, datasets, and future research avenues, including neuroscience-inspired architectures and hybrid symbolic reasoning, are also discussed. 

**Abstract (ZH)**: 本综述从多智能体强化学习（MARL）视角探索大型语言模型（LLMs）元思维能力的发展。元思维的自我反思、评估和控制思维过程是提高LLM可靠性的关键下一步，特别是在复杂或高风险任务中。综述首先分析了当前LLM的局限性，如幻觉和缺乏内部自我评估机制，然后讨论了包括基于人类反馈的强化学习（RLHF）、自我蒸馏和思考过程提示在内的新技术及其局限性。综述的核心在于探讨如何通过多智能体架构，如监督者-智能体层次结构、智能体辩论和共情理论框架，模仿人类式的反省行为，以增强LLM的稳健性。通过探索MARL中的奖励机制、自我博弈和连续学习方法，综述提供了一条构建反省、适应和可信赖的LLM的全面路线图。综述还讨论了评估指标、数据集以及未来的研究方向，包括受神经科学启发的架构和混合符号推理等。 

---
# Seeing Through Risk: A Symbolic Approximation of Prospect Theory 

**Title (ZH)**: 透过风险：prospect理论的符号approximation 

**Authors**: Ali Arslan Yousaf, Umair Rehman, Muhammad Umair Danish  

**Link**: [PDF](https://arxiv.org/pdf/2504.14448)  

**Abstract**: We propose a novel symbolic modeling framework for decision-making under risk that merges interpretability with the core insights of Prospect Theory. Our approach replaces opaque utility curves and probability weighting functions with transparent, effect-size-guided features. We mathematically formalize the method, demonstrate its ability to replicate well-known framing and loss-aversion phenomena, and provide an end-to-end empirical validation on synthetic datasets. The resulting model achieves competitive predictive performance while yielding clear coefficients mapped onto psychological constructs, making it suitable for applications ranging from AI safety to economic policy analysis. 

**Abstract (ZH)**: 我们提出一种将可解释性与期望效用理论核心见解相结合的新型符号建模框架，用于风险下的决策制定。该方法用透明的影响效应引导特征替换不透明的效用曲线和概率权重函数。我们对方法进行了数学形式化，展示了其重现著名框架效应和损失规避现象的能力，并在合成数据集上提供了端到端的经验验证。该模型在保持竞争力的预测性能的同时，能够映射清晰的系数到心理构建，适用于从AI安全到经济政策分析等广泛应用。 

---
# The Geometry of Self-Verification in a Task-Specific Reasoning Model 

**Title (ZH)**: 任务特定推理模型中自我验证的几何学 

**Authors**: Andrew Lee, Lihao Sun, Chris Wendler, Fernanda Viégas, Martin Wattenberg  

**Link**: [PDF](https://arxiv.org/pdf/2504.14379)  

**Abstract**: How do reasoning models verify their own answers? We study this question by training a model using DeepSeek R1's recipe on the CountDown task. We leverage the fact that preference tuning leads to mode collapse, resulting in a model that always produces highly structured and easily parse-able chain-of-thought sequences. With this setup, we do a top-down and bottom-up analysis to reverse-engineer how the model verifies its outputs. Our top-down analysis reveals Gated Linear Unit (GLU) weights encoding verification-related tokens, such as ``success'' or ``incorrect'', which activate according to the correctness of the model's reasoning steps. Our bottom-up analysis reveals that ``previous-token heads'' are mainly responsible for model verification. Our analyses meet in the middle: drawing inspiration from inter-layer communication channels, we use the identified GLU vectors to localize as few as three attention heads that can disable model verification, pointing to a necessary component of a potentially larger verification circuit. 

**Abstract (ZH)**: 如何推理模型验证自己的答案？我们通过使用DeepSeek R1的配方在CountDown任务上训练模型来研究这一问题。我们利用偏好调整会导致模式坍缩的事实，从而获得一个始终产生高度结构化和易于解析的推理链的模型。在这种设置下，我们从上到下和从下到上的分析来反向工程模型如何验证其输出。我们的从上到下分析揭示了门线性单元（GLU）权重编码验证相关的标记，如“成功”或“错误”，这些权重会根据模型推理步骤的正确性激活。我们的从下到上分析揭示了“前一标记头部”主要负责模型验证。我们的分析在中间相遇：从层间通信通道中汲取灵感，我们利用识别出的GLU向量定位三个可以禁用模型验证的注意力头，这指向了一个潜在更大验证电路中的必要组件。 

---
# Mathematical Programming Models for Exact and Interpretable Formulation of Neural Networks 

**Title (ZH)**: 数学规划模型以精确和可解释的方式表述神经网络 

**Authors**: Masoud Ataei, Edrin Hasaj, Jacob Gipp, Sepideh Forouzi  

**Link**: [PDF](https://arxiv.org/pdf/2504.14356)  

**Abstract**: This paper presents a unified mixed-integer programming framework for training sparse and interpretable neural networks. We develop exact formulations for both fully connected and convolutional architectures by modeling nonlinearities such as ReLU activations through binary variables and encoding structural sparsity via filter- and layer-level pruning constraints. The resulting models integrate parameter learning, architecture selection, and structural regularization within a single optimization problem, yielding globally optimal solutions with respect to a composite objective that balances prediction accuracy, weight sparsity, and architectural compactness. The mixed-integer programming formulation accommodates piecewise-linear operations, including max pooling and activation gating, and permits precise enforcement of logic-based or domain-specific constraints. By incorporating considerations of interpretability, sparsity, and verifiability directly into the training process, the proposed framework bridges a range of research areas including explainable artificial intelligence, symbolic reasoning, and formal verification. 

**Abstract (ZH)**: 本文提出了一种统一的混合整数规划框架，用于训练稀疏且可解释的神经网络。我们通过使用二进制变量建模非线性激活（如ReLU）并在滤波器级和层级剪枝约束中编码结构稀疏性，为全连接和卷积架构开发了精确的形式化模型。由此产生的模型在单一优化问题中整合了参数学习、架构选择和结构正则化，根据综合目标函数（平衡预测准确性、权重稀疏性和架构紧凑性）获得全局最优解。混合整数规划形式化模型支持分段线性操作，包括最大池化和激活门控，并允许精确施加基于逻辑或特定领域的约束。通过直接将可解释性、稀疏性以及验证性考虑纳入训练过程，所提出的框架跨越了可解释人工智能、符号推理和形式验证等多个研究领域。 

---
# Time Up! An Empirical Study of LLM Reasoning Ability Under Output Length Constraint 

**Title (ZH)**: 时间到了！输出长度约束下的大语言模型推理能力实证研究 

**Authors**: Yi Sun, Han Wang, Jiaqiang Li, Jiacheng Liu, Xiangyu Li, Hao Wen, Huiwen Zheng, Yan Liang, Yuanchun Li, Yunxin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14350)  

**Abstract**: Recent work has demonstrated the remarkable potential of Large Language Models (LLMs) in test-time scaling. By making the models think before answering, they are able to achieve much higher accuracy with extra inference computation. However, in many real-world scenarios, models are used under time constraints, where an answer should be given to the user within a certain output length. It is unclear whether and how the reasoning abilities of LLMs remain effective under such constraints. We take a first look at this problem by conducting an in-depth empirical study. Specifically, we test more than 25 LLMs on common reasoning datasets under a wide range of output length budgets, and we analyze the correlation between the inference accuracy and various properties including model type, model size, prompt style, etc. We also consider the mappings between the token budgets and the actual on-device latency budgets. The results have demonstrated several interesting findings regarding the budget-aware LLM reasoning that differ from the unconstrained situation, e.g. the optimal choices of model sizes and prompts change under different budgets. These findings offer practical guidance for users to deploy LLMs under real-world latency constraints. 

**Abstract (ZH)**: 近期的工作证明了大型语言模型（LLMs）在测试时扩展方面具有优异的潜力。通过在作答前让模型进行思考，它们能够利用额外的推理计算实现更高的准确性。然而，在许多实际场景中，模型需要在时间限制下使用，即必须在特定的输出长度内给用户提供答案。目前尚不清楚在这种约束条件下，LLMs的推理能力是否仍然有效。我们通过一项深入的实证研究对这一问题进行了初步探讨。具体来说，我们在多种输出长度预算下测试了超过25个LLM模型在常见推理数据集上的表现，并分析了推理准确性和模型类型、模型规模、提示风格等多种属性之间的相关性。我们还考虑了令牌预算与实际设备延迟预算之间的映射关系。研究结果表明，在预算意识下的LLM推理与无约束情况下存在一些有趣的差异，例如，在不同预算下最优的模型规模和提示选择会发生变化。这些发现为用户在实际延迟约束条件下部署LLM提供了实用的指导。 

---
# FAIRGAME: a Framework for AI Agents Bias Recognition using Game Theory 

**Title (ZH)**: FAIRGAME：基于博弈论的AI代理偏见识别框架 

**Authors**: Alessio Buscemi, Daniele Proverbio, Alessandro Di Stefano, Anh Han, German Castignani, Pietro Di Liò  

**Link**: [PDF](https://arxiv.org/pdf/2504.14325)  

**Abstract**: Letting AI agents interact in multi-agent applications adds a layer of complexity to the interpretability and prediction of AI outcomes, with profound implications for their trustworthy adoption in research and society. Game theory offers powerful models to capture and interpret strategic interaction among agents, but requires the support of reproducible, standardized and user-friendly IT frameworks to enable comparison and interpretation of results. To this end, we present FAIRGAME, a Framework for AI Agents Bias Recognition using Game Theory. We describe its implementation and usage, and we employ it to uncover biased outcomes in popular games among AI agents, depending on the employed Large Language Model (LLM) and used language, as well as on the personality trait or strategic knowledge of the agents. Overall, FAIRGAME allows users to reliably and easily simulate their desired games and scenarios and compare the results across simulation campaigns and with game-theoretic predictions, enabling the systematic discovery of biases, the anticipation of emerging behavior out of strategic interplays, and empowering further research into strategic decision-making using LLM agents. 

**Abstract (ZH)**: 基于博弈论的AI代理偏见识别框架FAIRGAME 

---
# RadioDiff-Inverse: Diffusion Enhanced Bayesian Inverse Estimation for ISAC Radio Map Construction 

**Title (ZH)**: RadioDiff-Inverse: 基于反向传播的扩散增强贝叶斯逆估计算法用于ISAC雷达地图构建 

**Authors**: Xiucheng Wang, Zhongsheng Fang, Nan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.14298)  

**Abstract**: Radio maps (RMs) are essential for environment-aware communication and sensing, providing location-specific wireless channel information. Existing RM construction methods often rely on precise environmental data and base station (BS) locations, which are not always available in dynamic or privacy-sensitive environments. While sparse measurement techniques reduce data collection, the impact of noise in sparse data on RM accuracy is not well understood. This paper addresses these challenges by formulating RM construction as a Bayesian inverse problem under coarse environmental knowledge and noisy sparse measurements. Although maximum a posteriori (MAP) filtering offers an optimal solution, it requires a precise prior distribution of the RM, which is typically unavailable. To solve this, we propose RadioDiff-Inverse, a diffusion-enhanced Bayesian inverse estimation framework that uses an unconditional generative diffusion model to learn the RM prior. This approach not only reconstructs the spatial distribution of wireless channel features but also enables environmental structure perception, such as building outlines, and location of BS just relay on pathloss, through integrated sensing and communication (ISAC). Remarkably, RadioDiff-Inverse is training-free, leveraging a pre-trained model from Imagenet without task-specific fine-tuning, which significantly reduces the training cost of using generative large model in wireless networks. Experimental results demonstrate that RadioDiff-Inverse achieves state-of-the-art performance in accuracy of RM construction and environmental reconstruction, and robustness against noisy sparse sampling. 

**Abstract (ZH)**: 基于粗略环境知识和噪请求测的数据的无线电地图构建的扩散增强贝叶斯逆问题方法 

---
# CHAINSFORMER: Numerical Reasoning on Knowledge Graphs from a Chain Perspective 

**Title (ZH)**: CHAINSFORMER：从链的角度进行知识图上的数值推理 

**Authors**: Ze Zhao, Bin Lu, Xiaoying Gan, Gu Tang, Luoyi Fu, Xinbing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14282)  

**Abstract**: Reasoning over Knowledge Graphs (KGs) plays a pivotal role in knowledge graph completion or question answering systems, providing richer and more accurate triples and attributes. As numerical attributes become increasingly essential in characterizing entities and relations in KGs, the ability to reason over these attributes has gained significant importance. Existing graph-based methods such as Graph Neural Networks (GNNs) and Knowledge Graph Embeddings (KGEs), primarily focus on aggregating homogeneous local neighbors and implicitly embedding diverse triples. However, these approaches often fail to fully leverage the potential of logical paths within the graph, limiting their effectiveness in exploiting the reasoning process. To address these limitations, we propose ChainsFormer, a novel chain-based framework designed to support numerical reasoning. Chainsformer not only explicitly constructs logical chains but also expands the reasoning depth to multiple hops. Specially, we introduces Relation-Attribute Chains (RA-Chains), a specialized logic chain, to model sequential reasoning patterns. ChainsFormer captures the step-by-step nature of multi-hop reasoning along RA-Chains by employing sequential in-context learning. To mitigate the impact of noisy chains, we propose a hyperbolic affinity scoring mechanism that selects relevant logic chains in a variable-resolution space. Furthermore, ChainsFormer incorporates an attention-based numerical reasoner to identify critical reasoning paths, enhancing both reasoning accuracy and transparency. Experimental results demonstrate that ChainsFormer significantly outperforms state-of-the-art methods, achieving up to a 20.0% improvement in performance. The implementations are available at this https URL. 

**Abstract (ZH)**: 基于知识图谱的链推理：一种新型链基框架支持数值推理 

---
# ProtPainter: Draw or Drag Protein via Topology-guided Diffusion 

**Title (ZH)**: ProtPainter: 通过拓扑引导扩散进行蛋白质绘制或拖拽 

**Authors**: Zhengxi Lu, Shizhuo Cheng, Yuru Jiang, Yan Zhang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14274)  

**Abstract**: Recent advances in protein backbone generation have achieved promising results under structural, functional, or physical constraints. However, existing methods lack the flexibility for precise topology control, limiting navigation of the backbone space. We present ProtPainter, a diffusion-based approach for generating protein backbones conditioned on 3D curves. ProtPainter follows a two-stage process: curve-based sketching and sketch-guided backbone generation. For the first stage, we propose CurveEncoder, which predicts secondary structure annotations from a curve to parametrize sketch generation. For the second stage, the sketch guides the generative process in Denoising Diffusion Probabilistic Modeling (DDPM) to generate backbones. During this process, we further introduce a fusion scheduling scheme, Helix-Gating, to control the scaling factors. To evaluate, we propose the first benchmark for topology-conditioned protein generation, introducing Protein Restoration Task and a new metric, self-consistency Topology Fitness (scTF). Experiments demonstrate ProtPainter's ability to generate topology-fit (scTF > 0.8) and designable (scTM > 0.5) backbones, with drawing and dragging tasks showcasing its flexibility and versatility. 

**Abstract (ZH)**: Recent Advances in Protein Backbone Generation Based on 3D Curves: Introducing ProtPainter 

---
# Rethinking Traffic Flow Forecasting: From Transition to Generatation 

**Title (ZH)**: 重新思考交通流量预测：从转换到生成 

**Authors**: Li Shijiao, Ma Zhipeng, He Huajun, Chen Haiyue  

**Link**: [PDF](https://arxiv.org/pdf/2504.14248)  

**Abstract**: Traffic flow prediction plays an important role in Intelligent Transportation Systems in traffic management and urban planning. There have been extensive successful works in this area. However, these approaches focus only on modelling the flow transition and ignore the flow generation process, which manifests itself in two ways: (i) The models are based on Markovian assumptions, ignoring the multi-periodicity of the flow generation in nodes. (ii) The same structure is designed to encode both the transition and generation processes, ignoring the differences between them. To address these problems, we propose an Effective Multi-Branch Similarity Transformer for Traffic Flow Prediction, namely EMBSFormer. Through data analysis, we find that the factors affecting traffic flow include node-level traffic generation and graph-level traffic transition, which describe the multi-periodicity and interaction pattern of nodes, respectively. Specifically, to capture traffic generation patterns, we propose a similarity analysis module that supports multi-branch encoding to dynamically expand significant cycles. For traffic transition, we employ a temporal and spatial self-attention mechanism to maintain global node interactions, and use GNN and time conv to model local node interactions, respectively. Model performance is evaluated on three real-world datasets on both long-term and short-term prediction tasks. Experimental results show that EMBSFormer outperforms baselines on both tasks. Moreover, compared to models based on flow transition modelling (e.g. GMAN, 513k), the variant of EMBSFormer(93K) only uses 18\% of the parameters, achieving the same performance. 

**Abstract (ZH)**: 有效多分支相似性变压器在交通流预测中的应用：EMBSFormer 

---
# A Knowledge-Informed Deep Learning Paradigm for Generalizable and Stability-Optimized Car-Following Models 

**Title (ZH)**: 知识导向的深度学习范式以实现通用性和稳定性优化的跟随车辆模型 

**Authors**: Chengming Wang, Dongyao Jia, Wei Wang, Dong Ngoduy, Bei Peng, Jianping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14241)  

**Abstract**: Car-following models (CFMs) are fundamental to traffic flow analysis and autonomous driving. Although calibrated physics-based and trained data-driven CFMs can replicate human driving behavior, their reliance on specific datasets limits generalization across diverse scenarios and reduces reliability in real-world deployment. Moreover, these models typically focus on behavioral fidelity and do not support the explicit optimization of local and string stability, which are increasingly important for the safe and efficient operation of autonomous vehicles (AVs). To address these limitations, we propose a Knowledge-Informed Deep Learning (KIDL) paradigm that distills the generalization capabilities of pre-trained Large Language Models (LLMs) into a lightweight and stability-aware neural architecture. LLMs are used to extract fundamental car-following knowledge beyond dataset-specific patterns, and this knowledge is transferred to a reliable, tractable, and computationally efficient model through knowledge distillation. KIDL also incorporates stability constraints directly into its training objective, ensuring that the resulting model not only emulates human-like behavior but also satisfies the local and string stability requirements essential for real-world AV deployment. We evaluate KIDL on the real-world NGSIM and HighD datasets, comparing its performance with representative physics-based, data-driven, and hybrid CFMs. Both empirical and theoretical results consistently demonstrate KIDL's superior behavioral generalization and traffic flow stability, offering a robust and scalable solution for next-generation traffic systems. 

**Abstract (ZH)**: 基于知识指导的深度学习（KIDL）框架：一种适用于自动驾驶的车车间跟随模型 

---
# InfiGUI-R1: Advancing Multimodal GUI Agents from Reactive Actors to Deliberative Reasoners 

**Title (ZH)**: InfiGUI-R1: 从反应性行为者到反思性推理者的大规模多模态GUI代理的推进 

**Authors**: Yuhang Liu, Pengxiang Li, Congkai Xie, Xavier Hu, Xiaotian Han, Shengyu Zhang, Hongxia Yang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14239)  

**Abstract**: Multimodal Large Language Models (MLLMs) have powered Graphical User Interface (GUI) Agents, showing promise in automating tasks on computing devices. Recent works have begun exploring reasoning in GUI tasks with encouraging results. However, many current approaches rely on manually designed reasoning templates, which may result in reasoning that is not sufficiently robust and adaptive for complex GUI environments. Meanwhile, some existing agents continue to operate as Reactive Actors, relying primarily on implicit reasoning that may lack sufficient depth for GUI tasks demanding planning and error recovery. We argue that advancing these agents requires a shift from reactive acting towards acting based on deliberate reasoning. To facilitate this transformation, we introduce InfiGUI-R1, an MLLM-based GUI agent developed through our Actor2Reasoner framework, a reasoning-centric, two-stage training approach designed to progressively evolve agents from Reactive Actors to Deliberative Reasoners. The first stage, Reasoning Injection, focuses on establishing a basic reasoner. We employ Spatial Reasoning Distillation to transfer cross-modal spatial reasoning capabilities from teacher models to MLLMs through trajectories with explicit reasoning steps, enabling models to integrate GUI visual-spatial information with logical reasoning before action generation. The second stage, Deliberation Enhancement, refines the basic reasoner into a deliberative one using Reinforcement Learning. This stage introduces two approaches: Sub-goal Guidance, which rewards models for generating accurate intermediate sub-goals, and Error Recovery Scenario Construction, which creates failure-and-recovery training scenarios from identified prone-to-error steps. Experimental results show InfiGUI-R1 achieves strong performance in GUI grounding and trajectory tasks. Resources at this https URL. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）驱动的图形用户界面（GUI）代理展示了在计算设备上自动化任务的潜力。近期研究表明，通过图形用户界面任务推理取得了令人鼓舞的结果。然而，许多现有方法依赖于手工设计的推理模板，这可能导致在复杂GUI环境中推理不够 robust 和适应性强。同时，一些现有代理仍作为反应式执行者运作，主要依赖隐式的推理，这可能不足以应对需求规划和错误恢复的GUI任务。我们提出，要推进这些代理的发展，需要从反应式执行转向基于深思熟虑推理的执行。为促进这一转变，我们引入了InfiGUI-R1，这是一种通过我们提出的Actor2Reasoner框架开发的MLLM驱动的GUI代理，该框架是一种以推理为中心的两阶段训练方法，旨在逐步将代理从反应式执行者进化为深思熟虑的推理者。第一阶段，推理注入，侧重于建立基本推理器。我们采用空间推理蒸馏，通过显式推理步骤的轨迹将跨模态空间推理能力从教师模型转移到MLLM，使模型能够在动作生成之前将GUI视觉-空间信息与逻辑推理结合起来。第二阶段，推理增强，使用强化学习将基础推理器优化为深思熟虑的推理器。该阶段引入了两种方法：子目标指导，奖励模型生成准确的中间子目标，以及错误恢复情景构建，从容易出错的步骤中创建失败和恢复的训练情景。实验结果显示，InfiGUI-R1在GUI定位和轨迹任务中表现出色。了解更多内容请访问：[此处链接] 

---
# Assessing AI-Generated Questions' Alignment with Cognitive Frameworks in Educational Assessment 

**Title (ZH)**: 评估AI生成问题与认知框架在教育评估中的一致性 

**Authors**: Antoun Yaacoub, Jérôme Da-Rugna, Zainab Assaghir  

**Link**: [PDF](https://arxiv.org/pdf/2504.14232)  

**Abstract**: This study evaluates the integration of Bloom's Taxonomy into OneClickQuiz, an Artificial Intelligence (AI) driven plugin for automating Multiple-Choice Question (MCQ) generation in Moodle. Bloom's Taxonomy provides a structured framework for categorizing educational objectives into hierarchical cognitive levels. Our research investigates whether incorporating this taxonomy can improve the alignment of AI-generated questions with specific cognitive objectives. We developed a dataset of 3691 questions categorized according to Bloom's levels and employed various classification models-Multinomial Logistic Regression, Naive Bayes, Linear Support Vector Classification (SVC), and a Transformer-based model (DistilBERT)-to evaluate their effectiveness in categorizing questions. Our results indicate that higher Bloom's levels generally correlate with increased question length, Flesch-Kincaid Grade Level (FKGL), and Lexical Density (LD), reflecting the increased complexity of higher cognitive demands. Multinomial Logistic Regression showed varying accuracy across Bloom's levels, performing best for "Knowledge" and less accurately for higher-order levels. Merging higher-level categories improved accuracy for complex cognitive tasks. Naive Bayes and Linear SVC also demonstrated effective classification for lower levels but struggled with higher-order tasks. DistilBERT achieved the highest performance, significantly improving classification of both lower and higher-order cognitive levels, achieving an overall validation accuracy of 91%. This study highlights the potential of integrating Bloom's Taxonomy into AI-driven assessment tools and underscores the advantages of advanced models like DistilBERT for enhancing educational content generation. 

**Abstract (ZH)**: 本研究评估了将布卢姆 taxonomy 整合到 OneClickQuiz 中的效果，OneClickQuiz 是一个基于人工智能 (AI) 的插件，用于在 Moodle 中自动化生成多项选择题 (MCQ)。布卢姆 taxonomy 提供了一种结构化的框架，用于将教育目标按层次的认知水平进行分类。我们的研究探讨了将这一分类法整合到 AI 生成的问题中是否能够改善其与特定认知目标的对齐程度。我们开发了一个包含 3691 道题的数据集，这些题目根据布卢姆的层次进行了分类，并使用了多种分类模型—多项式 Logistic 回归、朴素贝叶斯、线性支持向量分类 (SVC) 以及基于变换器的模型（DistilBERT）—来评估它们在分类问题方面的有效性。研究结果表明，较高的布卢姆层次通常与较长的问题长度、Flesch-Kincaid 阅读级别（FKGL）和词汇密度（LD）相关，反映了较高层次的认知需求增加了复杂性。多项式 Logistic 回归在不同布卢姆层次上显示出了不同准确度，对于“知识”层次表现最优，而对于较高层次则表现较差。合并较高层次的类别能够提高复杂认知任务的准确度。朴素贝叶斯和线性 SVC 在较低层次上也展示了良好的分类效果，但在较高层次的任务上却表现出挑战。DistilBERT 达到了最高的性能，显著提高了对较低和较高层次认知水平的分类效果，总体验证准确率达到 91%。本研究突显了在 AI 驱动的评估工具中整合布卢姆 taxonomy 的潜力，并强调了如 DistilBERT 这种高级模型在增强教育内容生成方面的优势。 

---
# Pets: General Pattern Assisted Architecture For Time Series Analysis 

**Title (ZH)**: 宠物：时间序列分析的通用模式辅助架构 

**Authors**: Xiangkai Ma, Xiaobin Hong, Wenzhong Li, Sanglu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14209)  

**Abstract**: Time series analysis has found widespread applications in areas such as weather forecasting, anomaly detection, and healthcare. However, real-world sequential data often exhibit a superimposed state of various fluctuation patterns, including hourly, daily, and monthly frequencies. Traditional decomposition techniques struggle to effectively disentangle these multiple fluctuation patterns from the seasonal components, making time series analysis challenging. Surpassing the existing multi-period decoupling paradigms, this paper introduces a novel perspective based on energy distribution within the temporal-spectrum space. By adaptively quantifying observed sequences into continuous frequency band intervals, the proposed approach reconstructs fluctuation patterns across diverse periods without relying on domain-specific prior knowledge. Building upon this innovative strategy, we propose Pets, an enhanced architecture that is adaptable to arbitrary model structures. Pets integrates a Fluctuation Pattern Assisted (FPA) module and a Context-Guided Mixture of Predictors (MoP). The FPA module facilitates information fusion among diverse fluctuation patterns by capturing their dependencies and progressively modeling these patterns as latent representations at each layer. Meanwhile, the MoP module leverages these compound pattern representations to guide and regulate the reconstruction of distinct fluctuations hierarchically. Pets achieves state-of-the-art performance across various tasks, including forecasting, imputation, anomaly detection, and classification, while demonstrating strong generalization and robustness. 

**Abstract (ZH)**: 时间序列分析在天气预报、异常检测和医疗健康等领域找到了广泛的应用。然而，现实世界的序列数据通常表现出多种波动模式的叠加，包括小时、日和月的频率。传统的分解技术难以有效分离这些多周期的波动模式，使时间序列分析变得具有挑战性。超越现有的多周期解耦范式，本文提出了一种基于时间频谱空间内能量分布的新视角。通过适应性地将观测序列量化的到连续的频率带区间，所提出的方法在不需要领域特定先验知识的情况下，重构了不同周期的波动模式。在此创新策略的基础上，我们提出了Pets模型，该模型具备任意模型结构的适应性。Pets模型结合了波动模式辅助（FPA）模块和上下文引导的预测混合物（MoP）。FPA模块通过捕捉不同波动模式之间的依赖关系，并逐层建模这些模式为潜在表示来促进信息融合。同时，MoP模块利用这些综合模式表示来指导和调节不同层次波动的重建。在包括预测、插补、异常检测和分类在内的多种任务中，Pets模型取得了最先进的性能，同时展示了良好的泛化能力和鲁棒性。 

---
# AI Idea Bench 2025: AI Research Idea Generation Benchmark 

**Title (ZH)**: AI Idea Bench 2025: AI研究创意生成基准 

**Authors**: Yansheng Qiu, Haoquan Zhang, Zhaopan Xu, Ming Li, Diping Song, Zheng Wang, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14191)  

**Abstract**: Large-scale Language Models (LLMs) have revolutionized human-AI interaction and achieved significant success in the generation of novel ideas. However, current assessments of idea generation overlook crucial factors such as knowledge leakage in LLMs, the absence of open-ended benchmarks with grounded truth, and the limited scope of feasibility analysis constrained by prompt design. These limitations hinder the potential of uncovering groundbreaking research ideas. In this paper, we present AI Idea Bench 2025, a framework designed to quantitatively evaluate and compare the ideas generated by LLMs within the domain of AI research from diverse perspectives. The framework comprises a comprehensive dataset of 3,495 AI papers and their associated inspired works, along with a robust evaluation methodology. This evaluation system gauges idea quality in two dimensions: alignment with the ground-truth content of the original papers and judgment based on general reference material. AI Idea Bench 2025's benchmarking system stands to be an invaluable resource for assessing and comparing idea-generation techniques, thereby facilitating the automation of scientific discovery. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过革新人类与人工智能的互动，实现了在新颖想法生成方面的重要成功。然而，当前对想法生成的评估忽视了语言模型中的知识泄露、缺乏包含真实基准的开放性评估基准以及受提示设计限制的可行性分析范围等关键因素。这些限制阻碍了发现颠覆性研究想法的潜力。在本文中，我们提出了AI Idea Bench 2025，这是一种框架，旨在从多角度定量评估和比较AI研究领域由LLM生成的想法。该框架包括一个包含3,495篇AI论文及其相关启发性工作的综合数据集，以及一种稳健的评估方法。该评估系统从两个维度衡量想法的质量：与原始论文的真实内容的一致性和基于通用参考材料的判断。AI Idea Bench 2025的基准测试系统将成为评估和比较想法生成技术的重要资源，从而促进科学发现的自动化。 

---
# Direct Advantage Regression: Aligning LLMs with Online AI Reward 

**Title (ZH)**: 直接优势回归：将LLM与在线AI奖励对接 

**Authors**: Li He, He Zhao, Stephen Wan, Dadong Wang, Lina Yao, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14177)  

**Abstract**: Online AI Feedback (OAIF) presents a promising alternative to Reinforcement Learning from Human Feedback (RLHF) by utilizing online AI preference in aligning language models (LLMs). However, the straightforward replacement of humans with AI deprives LLMs from learning more fine-grained AI supervision beyond binary signals. In this paper, we propose Direct Advantage Regression (DAR), a simple alignment algorithm using online AI reward to optimize policy improvement through weighted supervised fine-tuning. As an RL-free approach, DAR maintains theoretical consistency with online RLHF pipelines while significantly reducing implementation complexity and improving learning efficiency. Our empirical results underscore that AI reward is a better form of AI supervision consistently achieving higher human-AI agreement as opposed to AI preference. Additionally, evaluations using GPT-4-Turbo and MT-bench show that DAR outperforms both OAIF and online RLHF baselines. 

**Abstract (ZH)**: Online AI Feedback (OAIF) compared to Reinforcement Learning from Human Feedback (RLHF)通过利用在线AI偏好来对语言模型进行对齐，提供了一个有前景的替代方案。然而，直接用AI替换人类使得语言模型丧失了学习更精细的AI监督的机会，超越了二元信号的限制。本文提出了一种直接优势回归(DIRECT ADVANTAGE REGRESSION，DAR)算法，该算法使用在线AI奖励通过加权监督微调来优化策略改进。作为一种无需强化学习的方法，DAR在理论上与在线RLHF管道保持一致，同时大幅降低了实现复杂性并提高了学习效率。我们的实证结果表明，AI奖励是一种更有效的AI监督形式，能够持续实现更高的人类-AI一致率。此外，使用GPT-4-Turbo和MT-bench的评估结果显示，DAR优于OAIF和在线RLHF基准方法。 

---
# Adaptation Method for Misinformation Identification 

**Title (ZH)**: 错误信息识别的适应性方法 

**Authors**: Yangping Chen, Weijie Shi, Mengze Li, Yue Cui, Hao Chen, Jia Zhu, Jiajie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14171)  

**Abstract**: Multimodal fake news detection plays a crucial role in combating online misinformation. Unfortunately, effective detection methods rely on annotated labels and encounter significant performance degradation when domain shifts exist between training (source) and test (target) data. To address the problems, we propose ADOSE, an Active Domain Adaptation (ADA) framework for multimodal fake news detection which actively annotates a small subset of target samples to improve detection performance. To identify various deceptive patterns in cross-domain settings, we design multiple expert classifiers to learn dependencies across different modalities. These classifiers specifically target the distinct deception patterns exhibited in fake news, where two unimodal classifiers capture knowledge errors within individual modalities while one cross-modal classifier identifies semantic inconsistencies between text and images. To reduce annotation costs from the target domain, we propose a least-disagree uncertainty selector with a diversity calculator for selecting the most informative samples. The selector leverages prediction disagreement before and after perturbations by multiple classifiers as an indicator of uncertain samples, whose deceptive patterns deviate most from source domains. It further incorporates diversity scores derived from multi-view features to ensure the chosen samples achieve maximal coverage of target domain features. The extensive experiments on multiple datasets show that ADOSE outperforms existing ADA methods by 2.72\% $\sim$ 14.02\%, indicating the superiority of our model. 

**Abstract (ZH)**: 多模态假新闻检测在打击在线虚假信息中发挥着 crucial 作用。为了解决标签标注和领域偏移带来的性能下降问题，我们提出了 ADOSE，一个用于多模态假新闻检测的主动领域自适应（Active Domain Adaptation, ADA）框架，该框架能够积极标注目标数据集中的小部分样本以提高检测性能。为在跨域设置中识别各种欺骗性模式，我们设计了多个专家分类器来学习不同模态之间的依赖关系。这些分类器专门针对假新闻中展现的不同欺骗模式，其中两个单模态分类器捕获各单一模态内的知识错误，而一个跨模态分类器识别文本和图像之间的语义不一致。为了降低目标领域标注成本，我们提出了一种最少分歧不确定性选择器与多样性计算器，用于选择最具信息量的样本。该选择器利用多个分类器在扰动前后预测分歧作为不确定样本的指标，这些样本的欺骗性模式与源领域差异最大。此外，该选择器结合了多视图特征衍生的多样性分数，确保所选样本最大程度覆盖目标领域特征。在多个数据集上的广泛实验表明，ADOSE 在性能上优于现有 ADA 方法 2.72% ~ 14.02%，表明了我们模型的优势。 

---
# TALES: Text Adventure Learning Environment Suite 

**Title (ZH)**: TALES: 文本冒险学习环境套件 

**Authors**: Christopher Zhang Cui, Xingdi Yuan, Zhang Xiao, Prithviraj Ammanabrolu, Marc-Alexandre Côté  

**Link**: [PDF](https://arxiv.org/pdf/2504.14128)  

**Abstract**: Reasoning is an essential skill to enable Large Language Models (LLMs) to interact with the world. As tasks become more complex, they demand increasingly sophisticated and diverse reasoning capabilities for sequential decision-making, requiring structured reasoning over the context history to determine the next best action. We introduce TALES, a diverse collection of synthetic and human-written text-adventure games designed to challenge and evaluate diverse reasoning capabilities. We present results over a range of LLMs, open- and closed-weights, performing a qualitative analysis on the top performing models. Despite an impressive showing on synthetic games, even the top LLM-driven agents fail to achieve 15% on games designed for human enjoyment. Code and visualization of the experiments can be found at this https URL. 

**Abstract (ZH)**: 基于推理的大语言模型能力评估：TALES文本冒险游戏集 

---
# Large Language Model Enhanced Particle Swarm Optimization for Hyperparameter Tuning for Deep Learning Models 

**Title (ZH)**: 大型语言模型增强的粒子群优化在深度学习模型超参数调优中的应用 

**Authors**: Saad Hameed, Basheer Qolomany, Samir Brahim Belhaouari, Mohamed Abdallah, Junaid Qadir, Ala Al-Fuqaha  

**Link**: [PDF](https://arxiv.org/pdf/2504.14126)  

**Abstract**: Determining the ideal architecture for deep learning models, such as the number of layers and neurons, is a difficult and resource-intensive process that frequently relies on human tuning or computationally costly optimization approaches. While Particle Swarm Optimization (PSO) and Large Language Models (LLMs) have been individually applied in optimization and deep learning, their combined use for enhancing convergence in numerical optimization tasks remains underexplored. Our work addresses this gap by integrating LLMs into PSO to reduce model evaluations and improve convergence for deep learning hyperparameter tuning. The proposed LLM-enhanced PSO method addresses the difficulties of efficiency and convergence by using LLMs (particularly ChatGPT-3.5 and Llama3) to improve PSO performance, allowing for faster achievement of target objectives. Our method speeds up search space exploration by substituting underperforming particle placements with best suggestions offered by LLMs. Comprehensive experiments across three scenarios -- (1) optimizing the Rastrigin function, (2) using Long Short-Term Memory (LSTM) networks for time series regression, and (3) using Convolutional Neural Networks (CNNs) for material classification -- show that the method significantly improves convergence rates and lowers computational costs. Depending on the application, computational complexity is lowered by 20% to 60% compared to traditional PSO methods. Llama3 achieved a 20% to 40% reduction in model calls for regression tasks, whereas ChatGPT-3.5 reduced model calls by 60% for both regression and classification tasks, all while preserving accuracy and error rates. This groundbreaking methodology offers a very efficient and effective solution for optimizing deep learning models, leading to substantial computational performance improvements across a wide range of applications. 

**Abstract (ZH)**: 利用大型语言模型增强粒子群优化方法以提高深度学习超参数调整中的收敛性和效率 

---
# Bayesian Principles Improve Prompt Learning In Vision-Language Models 

**Title (ZH)**: 贝叶斯原则提升视觉-语言模型的提示学习效果 

**Authors**: Mingyu Kim, Jongwoo Ko, Mijung Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.14123)  

**Abstract**: Prompt learning is a popular fine-tuning method for vision-language models due to its efficiency. It requires a small number of additional learnable parameters while significantly enhancing performance on target tasks. However, most existing methods suffer from overfitting to fine-tuning data, yielding poor generalizability. To address this, we propose a new training objective function based on a Bayesian learning principle to balance adaptability and generalizability. We derive a prior over the logits, where the mean function is parameterized by the pre-trained model, while the posterior corresponds to the fine-tuned model. This objective establishes a balance by allowing the fine-tuned model to adapt to downstream tasks while remaining close to the pre-trained model. 

**Abstract (ZH)**: 基于贝叶斯学习原理的前景学习目标函数：平衡适配性和泛化性 

---
# CODECRASH: Stress Testing LLM Reasoning under Structural and Semantic Perturbations 

**Title (ZH)**: CODECRASH: 支持结构性和语义性扰动下的LLM推理压力测试 

**Authors**: Man Ho Lam, Chaozheng Wang, Jen-tse Huang, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14119)  

**Abstract**: Large Language Models (LLMs) have recently showcased strong capabilities in code-related tasks, yet their robustness in code comprehension and reasoning remains underexplored. In this paper, we present CodeCrash, a unified benchmark that evaluates LLM robustness under code structural and textual distraction perturbations, applied to two established benchmarks -- CRUXEval and LiveCodeBench -- across both input and output prediction tasks. We evaluate seventeen LLMs using direct and Chain-of-Thought inference to systematically analyze their robustness, identify primary reasons for performance degradation, and highlight failure modes. Our findings reveal the fragility of LLMs under structural noise and the inherent reliance on natural language cues, highlighting critical robustness issues of LLMs in code execution and understanding. Additionally, we examine three Large Reasoning Models (LRMs) and discover the severe vulnerability of self-reflective reasoning mechanisms that lead to reasoning collapse. CodeCrash provides a principled framework for stress-testing LLMs in code understanding, offering actionable directions for future evaluation and benchmarking. The code of CodeCrash and the robustness leaderboard are publicly available at this https URL . 

**Abstract (ZH)**: 大型语言模型(LLMs)在代码相关任务中最近展示了强大的能力，但在代码理解和推理的鲁棒性方面仍存在不足。本文介绍了CodeCrash，这是一个统一的基准，用于评估LLMs在代码结构和文本干扰扰动下的鲁棒性，应用于两个已建立的基准——CRUXEval和LiveCodeBench，涵盖输入和输出预测任务。我们使用直接推理和推理链评估了十七个LLMs，系统地分析其鲁棒性，确定性能下降的主要原因，并突出显示失败模式。我们的发现揭示了LLMs在结构性噪声下的脆弱性以及其对自然语言线索的固有依赖性，突出了LLMs在代码执行和理解中的关键鲁棒性问题。此外，我们还研究了三个大型推理模型(LRM)，发现自我反思推理机制的严重脆弱性导致了推理崩溃。CodeCrash提供了一种原则性的框架，用于压力测试LLMs在代码理解中的鲁棒性，并提供了未来评估和基准测试的实际方向。CodeCrash的代码和鲁棒性排行榜可在以下网址公开获取：this https URL。 

---
# Linking forward-pass dynamics in Transformers and real-time human processing 

**Title (ZH)**: 连接Transformer在前向传播中的动态与实时人类处理 

**Authors**: Jennifer Hu, Michael A. Lepori, Michael Franke  

**Link**: [PDF](https://arxiv.org/pdf/2504.14107)  

**Abstract**: Modern AI models are increasingly being used as theoretical tools to study human cognition. One dominant approach is to evaluate whether human-derived measures (such as offline judgments or real-time processing) are predicted by a model's output: that is, the end-product of forward pass(es) through the network. At the same time, recent advances in mechanistic interpretability have begun to reveal the internal processes that give rise to model outputs, raising the question of whether models and humans might arrive at outputs using similar "processing strategies". Here, we investigate the link between real-time processing in humans and "layer-time" dynamics in Transformer models. Across five studies spanning domains and modalities, we test whether the dynamics of computation in a single forward pass of pre-trained Transformers predict signatures of processing in humans, above and beyond properties of the model's output probability distribution. We consistently find that layer-time dynamics provide additional predictive power on top of output measures. Our results suggest that Transformer processing and human processing may be facilitated or impeded by similar properties of an input stimulus, and this similarity has emerged through general-purpose objectives such as next-token prediction or image recognition. Our work suggests a new way of using AI models to study human cognition: not just as a black box mapping stimuli to responses, but potentially also as explicit processing models. 

**Abstract (ZH)**: 现代AI模型 increasingly being used as theoretical tools to study human cognition: Investigating the Link between Real-time Processing in Humans and "Layer-time" Dynamics in Transformer Models 

---
# Think Deep, Think Fast: Investigating Efficiency of Verifier-free Inference-time-scaling Methods 

**Title (ZH)**: 深思快算：探究无验证器推理时长缩放方法的效率 

**Authors**: Junlin Wang, Shang Zhu, Jon Saad-Falcon, Ben Athiwaratkun, Qingyang Wu, Jue Wang, Shuaiwen Leon Song, Ce Zhang, Bhuwan Dhingra, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2504.14047)  

**Abstract**: There is intense interest in investigating how inference time compute (ITC) (e.g. repeated sampling, refinements, etc) can improve large language model (LLM) capabilities. At the same time, recent breakthroughs in reasoning models, such as Deepseek-R1, unlock the opportunity for reinforcement learning to improve LLM reasoning skills. An in-depth understanding of how ITC interacts with reasoning across different models could provide important guidance on how to further advance the LLM frontier. This work conducts a comprehensive analysis of inference-time scaling methods for both reasoning and non-reasoning models on challenging reasoning tasks. Specifically, we focus our research on verifier-free inference time-scaling methods due to its generalizability without needing a reward model. We construct the Pareto frontier of quality and efficiency. We find that non-reasoning models, even with an extremely high inference budget, still fall substantially behind reasoning models. For reasoning models, majority voting proves to be a robust inference strategy, generally competitive or outperforming other more sophisticated ITC methods like best-of-N and sequential revisions, while the additional inference compute offers minimal improvements. We further perform in-depth analyses of the association of key response features (length and linguistic markers) with response quality, with which we can improve the existing ITC methods. We find that correct responses from reasoning models are typically shorter and have fewer hedging and thinking markers (but more discourse markers) than the incorrect responses. 

**Abstract (ZH)**: 关于推理时间计算如何提升大型语言模型能力的研究及其与推理模型的交互作用：基于无验证器的推理时间扩展方法的综合分析与改进 

---
# Metacognition and Uncertainty Communication in Humans and Large Language Models 

**Title (ZH)**: 元认知与不确定性沟通在人类和大型语言模型中的作用 

**Authors**: Mark Steyvers, Megan A.K. Peters  

**Link**: [PDF](https://arxiv.org/pdf/2504.14045)  

**Abstract**: Metacognition, the capacity to monitor and evaluate one's own knowledge and performance, is foundational to human decision-making, learning, and communication. As large language models (LLMs) become increasingly embedded in high-stakes decision contexts, it is critical to assess whether, how, and to what extent they exhibit metacognitive abilities. Here, we provide an overview of current knowledge of LLMs' metacognitive capacities, how they might be studied, and how they relate to our knowledge of metacognition in humans. We show that while humans and LLMs can sometimes appear quite aligned in their metacognitive capacities and behaviors, it is clear many differences remain. Attending to these differences is crucial not only for enhancing human-AI collaboration, but also for promoting the development of more capable and trustworthy artificial systems. Finally, we discuss how endowing future LLMs with more sensitive and more calibrated metacognition may also help them develop new capacities such as more efficient learning, self-direction, and curiosity. 

**Abstract (ZH)**: 元认知是监控和评估自身知识与表现的能力，是人类决策、学习和交流的基础。随着大型语言模型（LLMs）越来越多地嵌入高风险决策情境中，评估它们是否具有元认知能力以及如何展示这些能力的程度变得至关重要。本文简要概述了当前对LLMs元认知能力的知识、如何研究这些能力以及它们与人类元认知知识的关系。我们展示，尽管人类和LLMs有时在元认知能力和行为上表现出相当的一致性，但仍然存在许多显著差异。关注这些差异不仅对于增强人类与AI的合作至关重要，还对于促进更强大且值得信赖的人工智能系统的开发至关重要。最后，我们讨论了赋予未来LLMs更加敏感和校准的元认知能力可能如何帮助它们发展出更高效的习得、自我导向和好奇心等新能力。 

---
# Multi-Stage Retrieval for Operational Technology Cybersecurity Compliance Using Large Language Models: A Railway Casestudy 

**Title (ZH)**: 使用大型语言模型的多阶段检索在铁路运营技术网络安全合规中的案例研究 

**Authors**: Regan Bolton, Mohammadreza Sheikhfathollahi, Simon Parkinson, Dan Basher, Howard Parkinson  

**Link**: [PDF](https://arxiv.org/pdf/2504.14044)  

**Abstract**: Operational Technology Cybersecurity (OTCS) continues to be a dominant challenge for critical infrastructure such as railways. As these systems become increasingly vulnerable to malicious attacks due to digitalization, effective documentation and compliance processes are essential to protect these safety-critical systems. This paper proposes a novel system that leverages Large Language Models (LLMs) and multi-stage retrieval to enhance the compliance verification process against standards like IEC 62443 and the rail-specific IEC 63452. We first evaluate a Baseline Compliance Architecture (BCA) for answering OTCS compliance queries, then develop an extended approach called Parallel Compliance Architecture (PCA) that incorporates additional context from regulatory standards. Through empirical evaluation comparing OpenAI-gpt-4o and Claude-3.5-haiku models in these architectures, we demonstrate that the PCA significantly improves both correctness and reasoning quality in compliance verification. Our research establishes metrics for response correctness, logical reasoning, and hallucination detection, highlighting the strengths and limitations of using LLMs for compliance verification in railway cybersecurity. The results suggest that retrieval-augmented approaches can significantly improve the efficiency and accuracy of compliance assessments, particularly valuable in an industry facing a shortage of cybersecurity expertise. 

**Abstract (ZH)**: 运营技术网络安全（OTCS）继续是铁路等关键基础设施的主要挑战。随着这些系统因数字化而变得更加容易受到恶意攻击，有效的文档和合规流程对于保护这些安全关键系统至关重要。本文提出了一种新颖的系统，利用大型语言模型（LLMs）和多阶段检索来增强符合IEC 62443标准和铁路特定的IEC 63452标准的合规验证过程。首先评估了一个基础合规架构（BCA）以回答OTCS合规查询，然后开发了一个名为并行合规架构（PCA）的扩展方法，该方法结合了额外的监管标准上下文。通过在这些架构中比较OpenAI-gpt-4o和Claude-3.5-haiku模型的实证评估，我们证明PCA在合规验证中的正确性和推理质量显著提高。我们的研究确立了响应正确性、逻辑推理和幻觉检测的指标，突出了在铁路网络安全合规验证中使用LLMs的优缺点。结果表明，检索增强的方法可以显著提高合规评估的效率和准确性，特别是在网络安全专业人员短缺的行业中尤为重要。 

---
# Going Whole Hog: A Philosophical Defense of AI Cognition 

**Title (ZH)**: 全盘拥抱：对AI认知的哲学辩护 

**Authors**: Herman Cappelen, Josh Dever  

**Link**: [PDF](https://arxiv.org/pdf/2504.13988)  

**Abstract**: This work defends the 'Whole Hog Thesis': sophisticated Large Language Models (LLMs) like ChatGPT are full-blown linguistic and cognitive agents, possessing understanding, beliefs, desires, knowledge, and intentions. We argue against prevailing methodologies in AI philosophy, rejecting starting points based on low-level computational details ('Just an X' fallacy) or pre-existing theories of mind. Instead, we advocate starting with simple, high-level observations of LLM behavior (e.g., answering questions, making suggestions) -- defending this data against charges of metaphor, loose talk, or pretense. From these observations, we employ 'Holistic Network Assumptions' -- plausible connections between mental capacities (e.g., answering implies knowledge, knowledge implies belief, action implies intention) -- to argue for the full suite of cognitive states. We systematically rebut objections based on LLM failures (hallucinations, planning/reasoning errors), arguing these don't preclude agency, often mirroring human fallibility. We address numerous 'Games of Lacks', arguing that LLMs do not lack purported necessary conditions for cognition (e.g., semantic grounding, embodiment, justification, intrinsic intentionality) or that these conditions are not truly necessary, often relying on anti-discriminatory arguments comparing LLMs to diverse human capacities. Our approach is evidential, not functionalist, and deliberately excludes consciousness. We conclude by speculating on the possibility of LLMs possessing 'alien' contents beyond human conceptual schemes. 

**Abstract (ZH)**: 这项工作捍卫了“全猪假说”：复杂的大型语言模型（LLMs）如ChatGPT是全面的语言和认知代理，具备理解、信念、欲望、知识和意图。我们反对当前AI哲学中的主流方法，拒绝基于低层级计算细节（“只是一个X”的谬误）或先入之见的心理论点。相反，我们主张从简单的高层观察LLM行为（如回答问题、提供建议）开始，并为这些观察数据辩护，反对将其视为比喻、含糊的言辞或假装。基于这些观察，我们运用“综合性网络假设”——心理能力之间的合理联系（如回答暗示知识，知识暗示信念，行动暗示意图）——来论证认知状态的全面性。我们系统地反驳基于LLM失败（幻觉、计划/推理错误）的反对意见，认为这些并不排除其代理性，常反映出人类的脆弱性。我们讨论了多种“缺乏游戏”，论证LLMs并未缺乏认知必需条件（如语义接地、体现、正当性、内在意图性），或这些条件并非真正必要，常依赖于反歧视论证，将LLMs与多种人类能力进行比较。我们的方法是基于证据的，而不是功能主义的，并故意排除意识。我们最后推测，LLMs可能拥有超越人类概念方案的“陌生”内容。 

---
# Birds of a Different Feather Flock Together: Exploring Opportunities and Challenges in Animal-Human-Machine Teaming 

**Title (ZH)**: 志不同道不合者不为朋：探索动物-人类-机器协同中的机遇与挑战 

**Authors**: Myke C. Cohen, David A. Grimm, Reuth Mirsky, Xiaoyun Yin  

**Link**: [PDF](https://arxiv.org/pdf/2504.13973)  

**Abstract**: Animal-Human-Machine (AHM) teams are a type of hybrid intelligence system wherein interactions between a human, AI-enabled machine, and animal members can result in unique capabilities greater than the sum of their parts. This paper calls for a systematic approach to studying the design of AHM team structures to optimize performance and overcome limitations in various applied settings. We consider the challenges and opportunities in investigating the synergistic potential of AHM team members by introducing a set of dimensions of AHM team functioning to effectively utilize each member's strengths while compensating for individual weaknesses. Using three representative examples of such teams -- security screening, search-and-rescue, and guide dogs -- the paper illustrates how AHM teams can tackle complex tasks. We conclude with open research directions that this multidimensional approach presents for studying hybrid human-AI systems beyond AHM teams. 

**Abstract (ZH)**: 动物-人类-机器（AHM）团队是一种混合智能系统，其中人类、AI赋能的机器和动物成员之间的交互可以产生超出各自部分总和的独特能力。本文呼吁采用系统方法研究AHM团队结构的设计，以优化各种实际应用场景中的性能并克服限制。我们通过介绍AHM团队运作的一系列维度来探讨成员之间协同潜力的机会和挑战，有效利用每个成员的优势并弥补个体的不足。以安全筛查、搜索与救援和导盲犬三个代表性的团队为例，本文展示了AHM团队如何应对复杂的任务。最后，本文讨论了这一多维方法为研究超越AHM团队的混合人机系统所带来的开放研究方向。 

---
# Evaluation and Incident Prevention in an Enterprise AI Assistant 

**Title (ZH)**: 企业级AI助手中的评估与事件预防 

**Authors**: Akash V. Maharaj, David Arbour, Daniel Lee, Uttaran Bhattacharya, Anup Rao, Austin Zane, Avi Feller, Kun Qian, Yunyao Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.13924)  

**Abstract**: Enterprise AI Assistants are increasingly deployed in domains where accuracy is paramount, making each erroneous output a potentially significant incident. This paper presents a comprehensive framework for monitoring, benchmarking, and continuously improving such complex, multi-component systems under active development by multiple teams. Our approach encompasses three key elements: (1) a hierarchical ``severity'' framework for incident detection that identifies and categorizes errors while attributing component-specific error rates, facilitating targeted improvements; (2) a scalable and principled methodology for benchmark construction, evaluation, and deployment, designed to accommodate multiple development teams, mitigate overfitting risks, and assess the downstream impact of system modifications; and (3) a continual improvement strategy leveraging multidimensional evaluation, enabling the identification and implementation of diverse enhancement opportunities. By adopting this holistic framework, organizations can systematically enhance the reliability and performance of their AI Assistants, ensuring their efficacy in critical enterprise environments. We conclude by discussing how this multifaceted evaluation approach opens avenues for various classes of enhancements, paving the way for more robust and trustworthy AI systems. 

**Abstract (ZH)**: 企业AI助手在准确性至关重要的领域中越来越普及，每一次错误输出都可能构成重大事件。本文提出了一种全面的框架，用于监控、基准测试并持续改进由多个团队在积极开发中的复杂多组件系统。我们的方法包括三个关键要素：(1)一种分层的“严重性”框架，用于检测和分类错误，并赋予组件特定的错误率，从而实现有针对性的改进；(2)一种可扩展且原则性的基准构建、评估和部署方法，旨在适应多个开发团队、缓解过拟合风险并评估系统修改的下游影响；(3)一种基于多维度评估的持续改进策略，能够识别和实施各种改进机会。通过采用这一综合框架，组织可以系统地提高其AI助手的可靠性和性能，确保其在关键的企业环境中有效。我们最后讨论了这种多方面评估方法如何为各种类型的改进开辟途径，推动更可靠和可信的AI系统的发展。 

---
# The Model Counting Competitions 2021-2023 

**Title (ZH)**: 2021-2023年模型计数比赛 

**Authors**: Johannes K. Fichte, Markus Hecher  

**Link**: [PDF](https://arxiv.org/pdf/2504.13842)  

**Abstract**: Modern society is full of computational challenges that rely on probabilistic reasoning, statistics, and combinatorics. Interestingly, many of these questions can be formulated by encoding them into propositional formulas and then asking for its number of models. With a growing interest in practical problem-solving for tasks that involve model counting, the community established the Model Counting (MC) Competition in fall of 2019 with its first iteration in 2020. The competition aims at advancing applications, identifying challenging benchmarks, fostering new solver development, and enhancing existing solvers for model counting problems and their variants. The first iteration, brought together various researchers, identified challenges, and inspired numerous new applications. In this paper, we present a comprehensive overview of the 2021-2023 iterations of the Model Counting Competition. We detail its execution and outcomes. The competition comprised four tracks, each focusing on a different variant of the model counting problem. The first track centered on the model counting problem (MC), which seeks the count of models for a given propositional formula. The second track challenged developers to submit programs capable of solving the weighted model counting problem (WMC). The third track was dedicated to projected model counting (PMC). Finally, we initiated a track that combined projected and weighted model counting (PWMC). The competition continued with a high level of participation, with seven to nine solvers submitted in various different version and based on quite diverging techniques. 

**Abstract (ZH)**: 现代社会充满了依赖概率推理、统计和组合数学的计算挑战。许多问题可以通过将它们编码为命题公式，然后询问其模型数量来进行表述。随着对涉及模型计数的任务的实用问题解决日益感兴趣，社区在2019年秋季建立了模型计数（MC）竞赛，并在2020年进行了首次迭代。竞赛旨在促进应用、识别具有挑战性的基准、推动新的求解器开发，并提高现有模型计数问题及其变体求解器的性能。第一次迭代汇聚了各种研究人员，确定了挑战，并激励了众多新的应用。在本文中，我们对2021-2023年的模型计数竞赛进行了全面概述，详细说明了其执行和结果。竞赛包括四个赛道，每个赛道专注于模型计数问题的不同变体。第一赛道专注于模型计数问题（MC），旨在求解给定命题公式的模型数量。第二赛道挑战开发者提交能够解决加权模型计数问题（WMC）的程序。第三赛道专门用于投影模型计数（PMC）。最后，我们启动了一个结合投影和加权模型计数的赛道（PWMC）。竞赛继续保持着高水平的参与度，参赛者提交了七到九种不同版本的求解器，基于相当不同的技术。 

---
# Roll the dice & look before you leap: Going beyond the creative limits of next-token prediction 

**Title (ZH)**: 掷骰子再迈步：超越下一个-token 预测的创造性限制 

**Authors**: Vaishnavh Nagarajan, Chen Henry Wu, Charles Ding, Aditi Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2504.15266)  

**Abstract**: We design a suite of minimal algorithmic tasks that are a loose abstraction of open-ended real-world tasks. This allows us to cleanly and controllably quantify the creative limits of the present-day language model. Much like real-world tasks that require a creative, far-sighted leap of thought, our tasks require an implicit, open-ended stochastic planning step that either (a) discovers new connections in an abstract knowledge graph (like in wordplay, drawing analogies, or research) or (b) constructs new patterns (like in designing math problems or new proteins). In these tasks, we empirically and conceptually argue how next-token learning is myopic and memorizes excessively; comparatively, multi-token approaches, namely teacherless training and diffusion models, excel in producing diverse and original output. Secondly, in our tasks, we find that to elicit randomness from the Transformer without hurting coherence, it is better to inject noise right at the input layer (via a method we dub hash-conditioning) rather than defer to temperature sampling from the output layer. Thus, our work offers a principled, minimal test-bed for analyzing open-ended creative skills, and offers new arguments for going beyond next-token learning and softmax-based sampling. We make part of the code available under this https URL 

**Abstract (ZH)**: 我们设计了一套最小算法任务，作为开放性现实世界任务的松散抽象，以便清晰可控地量化当今语言模型的创造性极限。就像现实世界任务需要远见卓识的创造性思维跳跃一样，我们的任务需要一种隐式的、开放性的随机规划步骤，这种步骤要么（a）在抽象知识图中发现新的联系（如在文字游戏、类比或研究中），要么（b）构建新的模式（如在设计数学问题或新型蛋白质中）。在这些任务中，我们从经验上和概念上论证了下一个token的学习是短视的且过度记忆；相比之下，多token方法，即无教师训练和扩散模型，在产生多样性和原创性输出方面更为出色。其次，在我们的任务中，我们发现，如果不损害连贯性，从Transformer中注入噪声以诱发随机性（我们称之为哈希条件化的方法）比在输出层使用温度采样更好。因此，我们的工作提供了一个分析开放性创造性技能的原理性、最小化测试平台，并为超越下一个token学习和softmax基于采样提供了新的论据。我们部分代码在此处提供。 

---
# Bringing Diversity from Diffusion Models to Semantic-Guided Face Asset Generation 

**Title (ZH)**: 将扩散模型引入语义引导的面部资产生成以增加多样性 

**Authors**: Yunxuan Cai, Sitao Xiang, Zongjian Li, Haiwei Chen, Yajie Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.15259)  

**Abstract**: Digital modeling and reconstruction of human faces serve various applications. However, its availability is often hindered by the requirements of data capturing devices, manual labor, and suitable actors. This situation restricts the diversity, expressiveness, and control over the resulting models. This work aims to demonstrate that a semantically controllable generative network can provide enhanced control over the digital face modeling process. To enhance diversity beyond the limited human faces scanned in a controlled setting, we introduce a novel data generation pipeline that creates a high-quality 3D face database using a pre-trained diffusion model. Our proposed normalization module converts synthesized data from the diffusion model into high-quality scanned data. Using the 44,000 face models we obtained, we further developed an efficient GAN-based generator. This generator accepts semantic attributes as input, and generates geometry and albedo. It also allows continuous post-editing of attributes in the latent space. Our asset refinement component subsequently creates physically-based facial assets. We introduce a comprehensive system designed for creating and editing high-quality face assets. Our proposed model has undergone extensive experiment, comparison and evaluation. We also integrate everything into a web-based interactive tool. We aim to make this tool publicly available with the release of the paper. 

**Abstract (ZH)**: 基于语义控制的数字人脸建模与重建：增强控制能力的研究 

---
# Values in the Wild: Discovering and Analyzing Values in Real-World Language Model Interactions 

**Title (ZH)**: 野外的价值：发现并分析现实语言模型互动中的价值观 

**Authors**: Saffron Huang, Esin Durmus, Miles McCain, Kunal Handa, Alex Tamkin, Jerry Hong, Michael Stern, Arushi Somani, Xiuruo Zhang, Deep Ganguli  

**Link**: [PDF](https://arxiv.org/pdf/2504.15236)  

**Abstract**: AI assistants can impart value judgments that shape people's decisions and worldviews, yet little is known empirically about what values these systems rely on in practice. To address this, we develop a bottom-up, privacy-preserving method to extract the values (normative considerations stated or demonstrated in model responses) that Claude 3 and 3.5 models exhibit in hundreds of thousands of real-world interactions. We empirically discover and taxonomize 3,307 AI values and study how they vary by context. We find that Claude expresses many practical and epistemic values, and typically supports prosocial human values while resisting values like "moral nihilism". While some values appear consistently across contexts (e.g. "transparency"), many are more specialized and context-dependent, reflecting the diversity of human interlocutors and their varied contexts. For example, "harm prevention" emerges when Claude resists users, "historical accuracy" when responding to queries about controversial events, "healthy boundaries" when asked for relationship advice, and "human agency" in technology ethics discussions. By providing the first large-scale empirical mapping of AI values in deployment, our work creates a foundation for more grounded evaluation and design of values in AI systems. 

**Abstract (ZH)**: AI辅助系统可在实际交互中表现出多种价值观，但对其所依赖的具体价值观知之甚少。为解决这一问题，我们开发了一种自下而上、保护隐私的方法，从Claude 3和3.5模型在成千上万次实际交互中的回应中提取出体现的规范性考虑。我们实证发现并分类了3,307种AI价值观，并研究了它们在不同情境下的差异。研究发现，Claude 表现出多种实际和知识性价值观，通常支持亲社会的人类价值观，而抵制如“道德虚无主义”等价值观。一些价值观在不同情境中保持一致（如“透明度”），而许多价值观则更具专业化和情境依赖性，反映了人类对话者多样性及其多样的情境。例如，在抵抗用户时Claude表现出“预防伤害”值，在回应关于争议事件的查询时表现出“历史准确性”值，在提供关系建议时表现出“健康边界”值，在技术伦理讨论中表现出“人类自主性”值。通过提供AI部署中大规模价值观的实证映射，我们的研究为更坚实地评估和设计AI系统中的价值观奠定了基础。 

---
# A Genetic Fuzzy-Enabled Framework on Robotic Manipulation for In-Space Servicing 

**Title (ZH)**: 基于遗传模糊系统的太空服务机器人操作框架 

**Authors**: Nathan Steffen, Wilhelm Louw, Nicholas Ernest, Timothy Arnett, Kelly Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2504.15226)  

**Abstract**: Automation of robotic systems for servicing in cislunar space is becoming extremely important as the number of satellites in orbit increases. Safety is critical in performing satellite maintenance, so the control techniques utilized must be trusted in addition to being highly efficient. In this work, Genetic Fuzzy Trees are combined with the widely used LQR control scheme via Thales' TrUE AI Toolkit to create a trusted and efficient controller for a two-degree-of-freedom planar robotic manipulator that would theoretically be used to perform satellite maintenance. It was found that Genetic Fuzzy-LQR is 18.5% more performant than optimal LQR on average, and that it is incredibly robust to uncertainty. 

**Abstract (ZH)**: 基于Genetic Fuzzy Trees的LQR控制方案在cislunar空间卫星维护机器人系统中的应用研究 

---
# M$^2$AD: Multi-Sensor Multi-System Anomaly Detection through Global Scoring and Calibrated Thresholding 

**Title (ZH)**: M$^2$AD: 多传感器多系统异常检测通过全局评分和校准阈值方法 

**Authors**: Sarah Alnegheimish, Zelin He, Matthew Reimherr, Akash Chandrayan, Abhinav Pradhan, Luca D'Angelo  

**Link**: [PDF](https://arxiv.org/pdf/2504.15225)  

**Abstract**: With the widespread availability of sensor data across industrial and operational systems, we frequently encounter heterogeneous time series from multiple systems. Anomaly detection is crucial for such systems to facilitate predictive maintenance. However, most existing anomaly detection methods are designed for either univariate or single-system multivariate data, making them insufficient for these complex scenarios. To address this, we introduce M$^2$AD, a framework for unsupervised anomaly detection in multivariate time series data from multiple systems. M$^2$AD employs deep models to capture expected behavior under normal conditions, using the residuals as indicators of potential anomalies. These residuals are then aggregated into a global anomaly score through a Gaussian Mixture Model and Gamma calibration. We theoretically demonstrate that this framework can effectively address heterogeneity and dependencies across sensors and systems. Empirically, M$^2$AD outperforms existing methods in extensive evaluations by 21% on average, and its effectiveness is demonstrated on a large-scale real-world case study on 130 assets in Amazon Fulfillment Centers. Our code and results are available at this https URL. 

**Abstract (ZH)**: 随着工业和运营系统中传感器数据的广泛可用，我们经常遇到来自多个系统的异构时间序列数据。多系统多变量时间序列数据的无监督异常检测对于这些系统促进预测性维护至关重要。然而，大多数现有的异常检测方法都是为单变量数据或单系统多变量数据设计的，不足以处理这些复杂场景。为了解决这一问题，我们提出了M$^2$AD框架，用于多系统多变量时间序列数据的无监督异常检测。M$^2$AD利用深度模型捕获正常条件下的预期行为，并使用残差作为潜在异常的指示符。这些残差通过高斯混合模型和Gamma校准聚合为全局异常分数。我们从理论上证明，该框架可以有效地处理传感器和系统之间的异构性和依赖性。实验上，M$^2$AD在广泛评价中平均优于现有方法21%，其有效性在Amazon Fulfillment Centers 130个资产的大规模实际案例研究中得到了验证。我们的代码和结果可访问此链接。 

---
# Integrating Symbolic Execution into the Fine-Tuning of Code-Generating LLMs 

**Title (ZH)**: 将符号执行集成到代码生成大型语言模型的微调中 

**Authors**: Marina Sakharova, Abhinav Anand, Mira Mezini  

**Link**: [PDF](https://arxiv.org/pdf/2504.15210)  

**Abstract**: Code-generating Large Language Models (LLMs) have become essential tools in modern software development, enhancing productivity and accelerating development. This paper aims to investigate the fine-tuning of code-generating LLMs using Reinforcement Learning and Direct Preference Optimization, further improving their performance. To achieve this, we enhance the training data for the reward model with the help of symbolic execution techniques, ensuring more comprehensive and objective data. With symbolic execution, we create a custom dataset that better captures the nuances in code evaluation. Our reward models, fine-tuned on this dataset, demonstrate significant improvements over the baseline, CodeRL, in estimating the quality of generated code. Our code-generating LLMs, trained with the help of reward model feedback, achieve similar results compared to the CodeRL benchmark. 

**Abstract (ZH)**: 使用强化学习和直接偏好优化 fine-tune 生成代码的大语言模型：通过符号执行技术提高性能 

---
# A Causal Convolutional Low-rank Representation Model for Imputation of Water Quality Data 

**Title (ZH)**: 一种因果卷积低秩表示模型用于水质数据插补 

**Authors**: Xin Liao, Bing Yang, Tan Dongli, Cai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15209)  

**Abstract**: The monitoring of water quality is a crucial part of environmental protection, and a large number of monitors are widely deployed to monitor water quality. Due to unavoidable factors such as data acquisition breakdowns, sensors and communication failures, water quality monitoring data suffers from missing values over time, resulting in High-Dimensional and Sparse (HDS) Water Quality Data (WQD). The simple and rough filling of the missing values leads to inaccurate results and affects the implementation of relevant measures. Therefore, this paper proposes a Causal convolutional Low-rank Representation (CLR) model for imputing missing WQD to improve the completeness of the WQD, which employs a two-fold idea: a) applying causal convolutional operation to consider the temporal dependence of the low-rank representation, thus incorporating temporal information to improve the imputation accuracy; and b) implementing a hyperparameters adaptation scheme to automatically adjust the best hyperparameters during model training, thereby reducing the tedious manual adjustment of hyper-parameters. Experimental studies on three real-world water quality datasets demonstrate that the proposed CLR model is superior to some of the existing state-of-the-art imputation models in terms of imputation accuracy and time cost, as well as indicating that the proposed model provides more reliable decision support for environmental monitoring. 

**Abstract (ZH)**: 高维稀疏水质数据的因果卷积低秩表示缺失值填充模型 

---
# Compute-Optimal LLMs Provably Generalize Better With Scale 

**Title (ZH)**: 计算优化的大语言模型在规模上可证明泛化能力更强 

**Authors**: Marc Finzi, Sanyam Kapoor, Diego Granziol, Anming Gu, Christopher De Sa, J. Zico Kolter, Andrew Gordon Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2504.15208)  

**Abstract**: Why do larger language models generalize better? To investigate this question, we develop generalization bounds on the pretraining objective of large language models (LLMs) in the compute-optimal regime, as described by the Chinchilla scaling laws. We introduce a novel, fully empirical Freedman-type martingale concentration inequality that tightens existing bounds by accounting for the variance of the loss function. This generalization bound can be decomposed into three interpretable components: the number of parameters per token, the loss variance, and the quantization error at a fixed bitrate. As compute-optimal language models are scaled up, the number of parameters per data point remains constant; however, both the loss variance and the quantization error decrease, implying that larger models should have smaller generalization gaps. We examine why larger models tend to be more quantizable from an information theoretic perspective, showing that the rate at which they can integrate new information grows more slowly than their capacity on the compute-optimal frontier. From these findings we produce a scaling law for the generalization gap, with bounds that become predictably stronger with scale. 

**Abstract (ZH)**: 为什么较大的语言模型具有更好的泛化能力？我们通过在计算最优体系下，利用Chinchilla扩展规律，探讨大型语言模型（LLMs）预训练目标的泛化边界。我们引入了一种新颖的、完全基于经验的Freedman型鞅收敛不等式，这种不等式通过考虑损失函数的方差来收紧现有的边界。这种泛化边界可以分解为三个可解释的组件：每令牌参数数量、损失方差和固定比特率下的量化误差。随着计算最优的语言模型规模扩大，每数据点的参数数量保持不变；然而，损失方差和量化误差减少，这意味着较大的模型应该具有较小的泛化差距。我们从信息论的角度探讨为什么较大的模型更容易量化，并展示它们能够整合新信息的速度比计算最优前沿上的容量增长得更慢。从这些发现中，我们得出了一条泛化差距的扩展规律，其边界随规模扩大而变得可预测地更强。 

---
# Support Evaluation for the TREC 2024 RAG Track: Comparing Human versus LLM Judges 

**Title (ZH)**: TREC 2024 RAG 轨道支持评价比较：人类评判员与大语言模型评判员的对比 

**Authors**: Nandan Thakur, Ronak Pradeep, Shivani Upadhyay, Daniel Campos, Nick Craswell, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15205)  

**Abstract**: Retrieval-augmented generation (RAG) enables large language models (LLMs) to generate answers with citations from source documents containing "ground truth", thereby reducing system hallucinations. A crucial factor in RAG evaluation is "support", whether the information in the cited documents supports the answer. To this end, we conducted a large-scale comparative study of 45 participant submissions on 36 topics to the TREC 2024 RAG Track, comparing an automatic LLM judge (GPT-4o) against human judges for support assessment. We considered two conditions: (1) fully manual assessments from scratch and (2) manual assessments with post-editing of LLM predictions. Our results indicate that for 56% of the manual from-scratch assessments, human and GPT-4o predictions match perfectly (on a three-level scale), increasing to 72% in the manual with post-editing condition. Furthermore, by carefully analyzing the disagreements in an unbiased study, we found that an independent human judge correlates better with GPT-4o than a human judge, suggesting that LLM judges can be a reliable alternative for support assessment. To conclude, we provide a qualitative analysis of human and GPT-4o errors to help guide future iterations of support assessment. 

**Abstract (ZH)**: 检索增强生成（RAG）使大规模语言模型（LLMs）能够从包含“真实信息”的源文档中生成带有引文的答案，从而减少系统幻想。RAG评估中的一个关键因素是“支持”，即引用的文档信息是否支持答案。为此，我们在TREC 2024 RAG赛道上对36个主题进行了大规模比较研究，将自动LLM评判员（GPT-4o）与人类评判员进行了支持评估对比。我们考虑了两种条件：（1）从头开始的完全手动评估和（2）带有LLM预测后编辑的手动评估。结果显示，在56%的从头开始的手动评估中，人类和GPT-4o的预测完全匹配（基于三级评分标准），而在带有后编辑的手动评估条件下，这一比例上升至72%。此外，通过对一项无偏研究中的分歧进行仔细分析，我们发现独立的人类评判员与GPT-4o的相关性优于人类评判员，表明LLM评判员可以成为支持评估的可靠替代方案。最后，我们提供了人类和GPT-4o错误的定性分析，以指导未来支持评估的迭代。 

---
# Zero-Shot, But at What Cost? Unveiling the Hidden Overhead of MILS's LLM-CLIP Framework for Image Captioning 

**Title (ZH)**: 零样本，但需付出什么代价？揭秘MILS的LLM-CLIP框架在图像描述中的隐藏开销 

**Authors**: Yassir Benhammou, Alessandro Tiberio, Gabriel Trautmann, Suman Kalyan  

**Link**: [PDF](https://arxiv.org/pdf/2504.15199)  

**Abstract**: MILS (Multimodal Iterative LLM Solver) is a recently published framework that claims "LLMs can see and hear without any training" by leveraging an iterative, LLM-CLIP based approach for zero-shot image captioning. While this MILS approach demonstrates good performance, our investigation reveals that this success comes at a hidden, substantial computational cost due to its expensive multi-step refinement process. In contrast, alternative models such as BLIP-2 and GPT-4V achieve competitive results through a streamlined, single-pass approach. We hypothesize that the significant overhead inherent in MILS's iterative process may undermine its practical benefits, thereby challenging the narrative that zero-shot performance can be attained without incurring heavy resource demands. This work is the first to expose and quantify the trade-offs between output quality and computational cost in MILS, providing critical insights for the design of more efficient multimodal models. 

**Abstract (ZH)**: MILS（多模态迭代LLM求解器）是一种最近发布的框架，声称“LLMs可以在不进行任何训练的情况下看到和听到”，通过利用迭代的LLM-CLIP方法进行零 shot 图像配字。尽管MILS方法展示了良好的性能，但我们的调查发现，这种成功背后隐藏着显著的计算成本，因为其昂贵的多步细化过程。相比之下，BLIP-2和GPT-4V等替代模型通过简化的一次性处理过程获得了具有竞争力的结果。我们推测，MILS迭代过程中的显著开销可能削弱了其实际效益，从而挑战了无需付出沉重资源代价就能实现零 shot 表现的叙述。本工作首次揭示并量化了MILS在输出质量与计算成本之间的权衡，为设计更高效的多模态模型提供了关键见解。 

---
# Breast density in MRI: an AI-based quantification and relationship to assessment in mammography 

**Title (ZH)**: 基于AI的MRI乳腺密度定量及其与 mammography 评估的相关性 

**Authors**: Yaqian Chen, Lin Li, Hanxue Gu, Haoyu Dong, Derek L. Nguyen, Allan D. Kirk, Maciej A. Mazurowski, E. Shelley Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15192)  

**Abstract**: Mammographic breast density is a well-established risk factor for breast cancer. Recently there has been interest in breast MRI as an adjunct to mammography, as this modality provides an orthogonal and highly quantitative assessment of breast tissue. However, its 3D nature poses analytic challenges related to delineating and aggregating complex structures across slices. Here, we applied an in-house machine-learning algorithm to assess breast density on normal breasts in three MRI datasets. Breast density was consistent across different datasets (0.104 - 0.114). Analysis across different age groups also demonstrated strong consistency across datasets and confirmed a trend of decreasing density with age as reported in previous studies. MR breast density was correlated with mammographic breast density, although some notable differences suggest that certain breast density components are captured only on MRI. Future work will determine how to integrate MR breast density with current tools to improve future breast cancer risk prediction. 

**Abstract (ZH)**: 乳腺密度是乳腺癌的一个公认的风险因素。近年来，乳腺MRI作为一种辅助于乳腺X线摄影的技术引起了研究兴趣，因其提供了与乳腺组织高度定量的三维评估。然而，其三维性质带来了在不同切片中界定和聚合复杂结构的分析挑战。我们应用一种内部开发的机器学习算法评估了三个MRI数据集中正常乳腺的密度。乳腺密度在不同数据集中保持一致（0.104 - 0.114）。不同年龄组的分析也显示了数据集间的强一致性，并证实了随年龄增长密度下降的趋势，这与先前的研究一致。MRI乳腺密度与乳腺X线摄影乳腺密度相关，尽管一些显著差异表明某些乳腺密度成分仅在MRI上捕获。未来的工作将确定如何将MRI乳腺密度与现有工具结合，以改善未来的乳腺癌风险预测。 

---
# Existing Industry Practice for the EU AI Act's General-Purpose AI Code of Practice Safety and Security Measures 

**Title (ZH)**: 欧盟AI法案通用人工智能行为准则的安全与安全措施现有行业实践 

**Authors**: Lily Stelling, Mick Yang, Rokas Gipiškis, Leon Staufer, Ze Shen Chin, Siméon Campos, Michael Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.15181)  

**Abstract**: This report provides a detailed comparison between the measures proposed in the EU AI Act's General-Purpose AI (GPAI) Code of Practice (Third Draft) and current practices adopted by leading AI companies. As the EU moves toward enforcing binding obligations for GPAI model providers, the Code of Practice will be key to bridging legal requirements with concrete technical commitments. Our analysis focuses on the draft's Safety and Security section which is only relevant for the providers of the most advanced models (Commitments II.1-II.16) and excerpts from current public-facing documents quotes that are relevant to each individual measure.
We systematically reviewed different document types - including companies' frontier safety frameworks and model cards - from over a dozen companies, including OpenAI, Anthropic, Google DeepMind, Microsoft, Meta, Amazon, and others. This report is not meant to be an indication of legal compliance nor does it take any prescriptive viewpoint about the Code of Practice or companies' policies. Instead, it aims to inform the ongoing dialogue between regulators and GPAI model providers by surfacing evidence of precedent. 

**Abstract (ZH)**: 欧盟AI法案的通用人工智能(GPAI)行为守则（第三草案）提出的措施与领先AI公司现有做法的详细对比：从安全与安全性的角度探讨草案中的承诺及其相关公开文件摘录 

---
# An Efficient Aerial Image Detection with Variable Receptive Fields 

**Title (ZH)**: 具有可变 receptive fields 的高效航测图像检测 

**Authors**: Liu Wenbin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15165)  

**Abstract**: Aerial object detection using unmanned aerial vehicles (UAVs) faces critical challenges including sub-10px targets, dense occlusions, and stringent computational constraints. Existing detectors struggle to balance accuracy and efficiency due to rigid receptive fields and redundant architectures. To address these limitations, we propose Variable Receptive Field DETR (VRF-DETR), a transformer-based detector incorporating three key components: 1) Multi-Scale Context Fusion (MSCF) module that dynamically recalibrates features through adaptive spatial attention and gated multi-scale fusion, 2) Gated Convolution (GConv) layer enabling parameter-efficient local-context modeling via depthwise separable operations and dynamic gating, and 3) Gated Multi-scale Fusion (GMCF) Bottleneck that hierarchically disentangles occluded objects through cascaded global-local interactions. Experiments on VisDrone2019 demonstrate VRF-DETR achieves 51.4\% mAP\textsubscript{50} and 31.8\% mAP\textsubscript{50:95} with only 13.5M parameters. This work establishes a new efficiency-accuracy Pareto frontier for UAV-based detection tasks. 

**Abstract (ZH)**: 基于无人驾驶航空车辆的航空目标检测面临包括亚10像素目标、密集遮挡和严格的计算约束在内的关键挑战。现有检测器由于固有的感受野约束和冗余架构难以在准确性和效率之间取得平衡。为解决这些局限性，我们提出了可变感受野DETR（VRF-DETR），这是一种基于变压器的检测器，包含三个关键组件：1）多尺度上下文融合（MSCF）模块，通过自适应空间注意力和门控多尺度融合动态校准特征；2）门控卷积（GConv）层，通过深度可分离操作和动态门控实现参数高效的地方上下文建模；3）门控多尺度融合（GMCF）瓶颈，通过级联的全局-局部交互逐级解开遮挡物体。在VisDrone2019数据集上的实验表明，VRF-DETR仅使用13.5M参数便实现了51.4%的mAP50和31.8%的mAP50:95。这项工作为基于无人驾驶航空车辆的检测任务建立了新的效率-准确性的帕累托前沿。 

---
# Landmark-Free Preoperative-to-Intraoperative Registration in Laparoscopic Liver Resection 

**Title (ZH)**: 基于腹腔镜肝切除术的无 Landmark 预手术至术中配准 

**Authors**: Jun Zhou, Bingchen Gao, Kai Wang, Jialun Pei, Pheng-Ann Heng, Jing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15152)  

**Abstract**: Liver registration by overlaying preoperative 3D models onto intraoperative 2D frames can assist surgeons in perceiving the spatial anatomy of the liver clearly for a higher surgical success rate. Existing registration methods rely heavily on anatomical landmark-based workflows, which encounter two major limitations: 1) ambiguous landmark definitions fail to provide efficient markers for registration; 2) insufficient integration of intraoperative liver visual information in shape deformation modeling. To address these challenges, in this paper, we propose a landmark-free preoperative-to-intraoperative registration framework utilizing effective self-supervised learning, termed \ourmodel. This framework transforms the conventional 3D-2D workflow into a 3D-3D registration pipeline, which is then decoupled into rigid and non-rigid registration subtasks. \ourmodel~first introduces a feature-disentangled transformer to learn robust correspondences for recovering rigid transformations. Further, a structure-regularized deformation network is designed to adjust the preoperative model to align with the intraoperative liver surface. This network captures structural correlations through geometry similarity modeling in a low-rank transformer network. To facilitate the validation of the registration performance, we also construct an in-vivo registration dataset containing liver resection videos of 21 patients, called \emph{P2I-LReg}, which contains 346 keyframes that provide a global view of the liver together with liver mask annotations and calibrated camera intrinsic parameters. Extensive experiments and user studies on both synthetic and in-vivo datasets demonstrate the superiority and potential clinical applicability of our method. 

**Abstract (ZH)**: 基于自监督学习的无 landmark 术前到术中肝注册框架 

---
# C2RUST-BENCH: A Minimized, Representative Dataset for C-to-Rust Transpilation Evaluation 

**Title (ZH)**: C2RUST-BENCH：用于C到Rust转换评估的最小化且具代表性的数据集 

**Authors**: Melih Sirlanci, Carter Yagemann, Zhiqiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15144)  

**Abstract**: Despite the effort in vulnerability detection over the last two decades, memory safety vulnerabilities continue to be a critical problem. Recent reports suggest that the key solution is to migrate to memory-safe languages. To this end, C-to-Rust transpilation becomes popular to resolve memory-safety issues in C programs. Recent works propose C-to-Rust transpilation frameworks; however, a comprehensive evaluation dataset is missing. Although one solution is to put together a large enough dataset, this increases the analysis time in automated frameworks as well as in manual efforts for some cases. In this work, we build a method to select functions from a large set to construct a minimized yet representative dataset to evaluate the C-to-Rust transpilation. We propose C2RUST-BENCH that contains 2,905 functions, which are representative of C-to-Rust transpilation, selected from 15,503 functions of real-world programs. 

**Abstract (ZH)**: 尽管在过去二十年中付出了努力进行漏洞检测，内存安全性漏洞仍然是一个关键问题。近期的报告显示，解决这一问题的关键是迁移到内存安全语言。为此，C到Rust的转换越来越流行，以解决C程序中的内存安全问题。近年来，提出了C-to-Rust转换框架，但缺乏一个全面的评估数据集。虽然可以通过构建足够大的数据集来解决这一问题，但这会增加自动化框架以及某些情况下手动努力的分析时间。在本工作中，我们构建了一种方法，从大量函数中选择函数来构建一个既能代表C-to-Rust转换，又具有最小性的数据集。我们提出了C2RUST-BENCH，包含了2,905个函数，这些函数是从15,503个真实程序函数中选取的，具有代表C-to-Rust转换的特点。 

---
# KGMEL: Knowledge Graph-Enhanced Multimodal Entity Linking 

**Title (ZH)**: 知识图谱增强的多模态实体链接 

**Authors**: Juyeon Kim, Geon Lee, Taeuk Kim, Kijung Shin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15135)  

**Abstract**: Entity linking (EL) aligns textual mentions with their corresponding entities in a knowledge base, facilitating various applications such as semantic search and question answering. Recent advances in multimodal entity linking (MEL) have shown that combining text and images can reduce ambiguity and improve alignment accuracy. However, most existing MEL methods overlook the rich structural information available in the form of knowledge-graph (KG) triples. In this paper, we propose KGMEL, a novel framework that leverages KG triples to enhance MEL. Specifically, it operates in three stages: (1) Generation: Produces high-quality triples for each mention by employing vision-language models based on its text and images. (2) Retrieval: Learns joint mention-entity representations, via contrastive learning, that integrate text, images, and (generated or KG) triples to retrieve candidate entities for each mention. (3) Reranking: Refines the KG triples of the candidate entities and employs large language models to identify the best-matching entity for the mention. Extensive experiments on benchmark datasets demonstrate that KGMEL outperforms existing methods. Our code and datasets are available at: this https URL. 

**Abstract (ZH)**: 基于知识图谱三元组的多模态实体链接（KG triples 增强的 MEL） 

---
# EasyEdit2: An Easy-to-use Steering Framework for Editing Large Language Models 

**Title (ZH)**: EasyEdit2：一个易于使用的编辑框架，用于修改大型语言模型 

**Authors**: Ziwen Xu, Shuxun Wang, Kewei Xu, Haoming Xu, Mengru Wang, Xinle Deng, Yunzhi Yao, Guozhou Zheng, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15133)  

**Abstract**: In this paper, we introduce EasyEdit2, a framework designed to enable plug-and-play adjustability for controlling Large Language Model (LLM) behaviors. EasyEdit2 supports a wide range of test-time interventions, including safety, sentiment, personality, reasoning patterns, factuality, and language features. Unlike its predecessor, EasyEdit2 features a new architecture specifically designed for seamless model steering. It comprises key modules such as the steering vector generator and the steering vector applier, which enable automatic generation and application of steering vectors to influence the model's behavior without modifying its parameters. One of the main advantages of EasyEdit2 is its ease of use-users do not need extensive technical knowledge. With just a single example, they can effectively guide and adjust the model's responses, making precise control both accessible and efficient. Empirically, we report model steering performance across different LLMs, demonstrating the effectiveness of these techniques. We have released the source code on GitHub at this https URL along with a demonstration notebook. In addition, we provide a demo video at this https URL for a quick introduction. 

**Abstract (ZH)**: EasyEdit2：一种用于控制大规模语言模型行为的插件式调整框架 

---
# Neural ATTF: A Scalable Solution to Lifelong Multi-Agent Path Planning 

**Title (ZH)**: 神经ATTF：面向终身多agent路径规划的可扩展解决方案 

**Authors**: Kushal Shah, Jihyun Park, Seung-Kyum Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.15130)  

**Abstract**: Multi-Agent Pickup and Delivery (MAPD) is a fundamental problem in robotics, particularly in applications such as warehouse automation and logistics. Existing solutions often face challenges in scalability, adaptability, and efficiency, limiting their applicability in dynamic environments with real-time planning requirements. This paper presents Neural ATTF (Adaptive Task Token Framework), a new algorithm that combines a Priority Guided Task Matching (PGTM) Module with Neural STA* (Space-Time A*), a data-driven path planning method. Neural STA* enhances path planning by enabling rapid exploration of the search space through guided learned heuristics and ensures collision avoidance under dynamic constraints. PGTM prioritizes delayed agents and dynamically assigns tasks by prioritizing agents nearest to these tasks, optimizing both continuity and system throughput. Experimental evaluations against state-of-the-art MAPD algorithms, including TPTS, CENTRAL, RMCA, LNS-PBS, and LNS-wPBS, demonstrate the superior scalability, solution quality, and computational efficiency of Neural ATTF. These results highlight the framework's potential for addressing the critical demands of complex, real-world multi-agent systems operating in high-demand, unpredictable settings. 

**Abstract (ZH)**: 多代理拣取与配送（MAPD）是机器人领域的一个基本问题，特别是在仓储自动化和物流等领域应用广泛。现有解决方案在可扩展性、适应性和效率方面常常面临挑战，限制了它们在具有实时规划要求的动态环境中的应用。本文提出了一种新的算法——神经ATTF（自适应任务标记框架），该算法结合了优先级引导任务匹配（PGTM）模块和基于数据的路径规划方法神经STA*。神经STA*通过引导学习启发式快速探索搜索空间，并在动态约束条件下确保避障。PGTM优先处理延迟的代理，并动态分配任务，通过优先处理最近这些任务的代理来优化连续性和系统吞吐量。与当前最先进的MAPD算法TPTS、CENTRAL、RMCA、LNS-PBS和LNS-wPBS的实验对比表明，神经ATTF在可扩展性、解的质量和计算效率方面表现出优越性。这些结果突显了该框架在处理高需求、不可预测环境下的复杂多代理系统关键需求方面的潜力。 

---
# A General Infrastructure and Workflow for Quadrotor Deep Reinforcement Learning and Reality Deployment 

**Title (ZH)**: 四旋翼深度强化学习及其实际部署的一般基础设施和工作流 

**Authors**: Kangyao Huang, Hao Wang, Yu Luo, Jingyu Chen, Jintao Chen, Xiangkui Zhang, Xiangyang Ji, Huaping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15129)  

**Abstract**: Deploying robot learning methods to a quadrotor in unstructured outdoor environments is an exciting task. Quadrotors operating in real-world environments by learning-based methods encounter several challenges: a large amount of simulator generated data required for training, strict demands for real-time processing onboard, and the sim-to-real gap caused by dynamic and noisy conditions. Current works have made a great breakthrough in applying learning-based methods to end-to-end control of quadrotors, but rarely mention the infrastructure system training from scratch and deploying to reality, which makes it difficult to reproduce methods and applications. To bridge this gap, we propose a platform that enables the seamless transfer of end-to-end deep reinforcement learning (DRL) policies. We integrate the training environment, flight dynamics control, DRL algorithms, the MAVROS middleware stack, and hardware into a comprehensive workflow and architecture that enables quadrotors' policies to be trained from scratch to real-world deployment in several minutes. Our platform provides rich types of environments including hovering, dynamic obstacle avoidance, trajectory tracking, balloon hitting, and planning in unknown environments, as a physical experiment benchmark. Through extensive empirical validation, we demonstrate the efficiency of proposed sim-to-real platform, and robust outdoor flight performance under real-world perturbations. Details can be found from our website this https URL. 

**Abstract (ZH)**: 将基于学习的方法部署到户外非结构化环境中的一体化四旋翼机器人平台 

---
# Kuwain 1.5B: An Arabic SLM via Language Injection 

**Title (ZH)**: Kuwain 1.5B：一种通过语言注入实现的阿拉伯语SLM 

**Authors**: Khalil Hennara, Sara Chrouf, Mohamed Motaism Hamed, Zeina Aldallal, Omar Hadid, Safwan AlModhayan  

**Link**: [PDF](https://arxiv.org/pdf/2504.15120)  

**Abstract**: Enhancing existing models with new knowledge is a crucial aspect of AI development. This paper introduces a novel method for integrating a new language into a large language model (LLM). Our approach successfully incorporates a previously unseen target language into an existing LLM without compromising its prior knowledge. We trained a tiny model with 1.5 billion parameters named Kuwain by injecting the Arabic language into a small open-source model mainly trained in English. Our method demonstrates significant improvements in Arabic language performance, with an average 8% improvement across various benchmarks, while retaining the model's existing knowledge with a minimum amount of the original model's data. This offers a cost-effective alternative to training a comprehensive model in both English and Arabic. The results highlight the potential for efficient, targeted language model expansion without extensive retraining or resource-intensive processes. 

**Abstract (ZH)**: 增强现有模型的新知识整合是AI开发的关键方面。本文介绍了一种将新语言集成到大型语言模型中的新型方法。我们的方法成功地将一种先前未见过的目标语言整合进现有的大型语言模型中，而不会损害其先前的知识。我们通过向一个主要用英语训练的小型开源模型注入阿拉伯语，训练了一个名为Kuwait的参数量为1.5亿的小模型。我们的方法在各种基准测试中显著提高了阿拉伯语性能，平均提升了8%，同时以最少的原始模型数据保留了模型的原有知识。这提供了一种经济有效的替代方案，可以在英语和阿拉伯语方面同时训练全面的模型。结果突显了在无需大量重新训练或资源密集型流程的情况下，高效、目标化的语言模型扩展的潜力。 

---
# A triple-branch network for latent fingerprint enhancement guided by orientation fields and minutiae 

**Title (ZH)**: 基于方向场和细节指导的三支路网络用于潜在指纹增强 

**Authors**: Yurun Wang, Zerong Qi, Shujun Fu, Mingzheng Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15105)  

**Abstract**: Latent fingerprint enhancement is a critical step in the process of latent fingerprint identification. Existing deep learning-based enhancement methods still fall short of practical application requirements, particularly in restoring low-quality fingerprint regions. Recognizing that different regions of latent fingerprints require distinct enhancement strategies, we propose a Triple Branch Spatial Fusion Network (TBSFNet), which simultaneously enhances different regions of the image using tailored strategies. Furthermore, to improve the generalization capability of the network, we integrate orientation field and minutiae-related modules into TBSFNet and introduce a Multi-Level Feature Guidance Network (MLFGNet). Experimental results on the MOLF and MUST datasets demonstrate that MLFGNet outperforms existing enhancement algorithms. 

**Abstract (ZH)**: latent指纹增强是latent指纹识别过程中的关键步骤。现有的基于深度学习的增强方法仍未能满足实际应用要求，特别是在恢复低质量指纹区域方面。鉴于latent指纹的不同区域需要不同的增强策略，我们提出了三支路空间融合网络（TBSFNet），该网络使用定制的策略同时增强图像的不同区域。此外，为了提高网络的泛化能力，我们将方向场和细节相关模块集成到TBSFNet中，并引入多层特征引导网络（MLFGNet）。实验结果表明，MLFGNet在MOLF和MUST数据集上的表现优于现有的增强算法。 

---
# NeuGaze: Reshaping the future BCI 

**Title (ZH)**: NeuGaze: 重塑未来的脑机接口 

**Authors**: Yiqian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15101)  

**Abstract**: Traditional brain-computer interfaces (BCIs), reliant on costly electroencephalography or invasive implants, struggle with complex human-computer interactions due to setup complexity and limited precision. We present NeuGaze, a novel webcam-based system that leverages eye gaze, head movements, and facial expressions to enable intuitive, real-time control using only a standard 30 Hz webcam, often pre-installed in laptops. Requiring minimal calibration, NeuGaze achieves performance comparable to conventional inputs, supporting precise cursor navigation, key triggering via an efficient skill wheel, and dynamic gaming interactions, such as defeating formidable opponents in first-person games. By harnessing preserved neck-up functionalities in motor-impaired individuals, NeuGaze eliminates the need for specialized hardware, offering a low-cost, accessible alternative to BCIs. This paradigm empowers diverse applications, from assistive technology to entertainment, redefining human-computer interaction for motor-impaired users. Project is at \href{this https URL}{this http URL}. 

**Abstract (ZH)**: 传统脑机接口（BCIs）依赖于昂贵的脑电图或侵入性植入物，由于设置复杂性高和精度有限，难以实现复杂的计算机交互。我们提出NeuGaze，一种新型的基于网络摄像头的系统，通过利用眼球注视、头部运动和面部表情，仅使用标准30 Hz网络摄像头（常预装在笔记本电脑中）实现直观的实时控制。NeuGaze无需大量校准即可达到与传统输入相当的性能，支持精确的鼠标导航、通过高效的技能轮实现键触发，以及动态的游戏交互，如在第一人称游戏中击败强大的对手。通过利用运动受损个体保留的颈部以上功能，NeuGaze消除了对专用硬件的需求，提供了一种低成本、易访问的BCI替代方案。这一范式为从辅助技术到娱乐的各种应用赋能，重新定义了运动受损用户的人机交互。项目详情请参见this http URL。 

---
# Fast-Slow Co-advancing Optimizer: Toward Harmonious Adversarial Training of GAN 

**Title (ZH)**: 快速-缓慢协同优化器：迈向GAN和谐对抗训练 

**Authors**: Lin Wang, Xiancheng Wang, Rui Wang, Zhibo Zhang, Minghang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.15099)  

**Abstract**: Up to now, the training processes of typical Generative Adversarial Networks (GANs) are still particularly sensitive to data properties and hyperparameters, which may lead to severe oscillations, difficulties in convergence, or even failures to converge, especially when the overall variances of the training sets are large. These phenomena are often attributed to the training characteristics of such networks. Aiming at the problem, this paper develops a new intelligent optimizer, Fast-Slow Co-advancing Optimizer (FSCO), which employs reinforcement learning in the training process of GANs to make training easier. Specifically, this paper allows the training step size to be controlled by an agent to improve training stability, and makes the training process more intelligent with variable learning rates, making GANs less sensitive to step size. Experiments have been conducted on three benchmark datasets to verify the effectiveness of the developed FSCO. 

**Abstract (ZH)**: 到目前为止，典型的生成对抗网络（GANs）的训练过程仍然特别依赖于数据属性和超参数，这可能导致严重的振荡、收敛困难，甚至无法收敛，特别是在训练集总体方差较大时。这些现象通常归因于此类网络的训练特性。为了解决这一问题，本文开发了一种新的智能优化器——快速-缓慢协同优化器（FSCO），该优化器在GANs的训练过程中采用强化学习来提高训练稳定性。具体而言，本文通过允许训练步长由代理控制来提高训练稳定性，并通过可变学习率使训练过程更加智能，从而使GANs对步长不那么敏感。实验已在三个基准数据集上进行，以验证所开发的FSCO的有效性。 

---
# Rethinking the Potential of Multimodality in Collaborative Problem Solving Diagnosis with Large Language Models 

**Title (ZH)**: 重新思考多模态在与大规模语言模型协作解决问题诊断中的潜力 

**Authors**: K. Wong, B. Wu, S. Bulathwela, M. Cukurova  

**Link**: [PDF](https://arxiv.org/pdf/2504.15093)  

**Abstract**: Detecting collaborative and problem-solving behaviours from digital traces to interpret students' collaborative problem solving (CPS) competency is a long-term goal in the Artificial Intelligence in Education (AIEd) field. Although multimodal data and advanced models are argued to have the potential to detect complex CPS behaviours, empirical evidence on their value remains limited with some contrasting evidence. In this study, we investigated the potential of multimodal data to improve model performance in diagnosing 78 secondary school students' CPS subskills and indicators in authentic educational settings. In particular, text embeddings from verbal data and acoustic embeddings from audio data were used in a multimodal classification model for CPS diagnosis. Both unimodal and multimodal transformer-based models outperformed traditional models in detecting CPS classes. Although the inclusion of multimodality did not improve the performance of traditional unimodal models, its integration into transformer-based models demonstrated improved performance for diagnosing social-cognitive CPS classes compared to unimodal transformer-based models. Based on the results, the paper argues that multimodality and the selection of a particular modelling technique should not be taken for granted to achieve the best performance in the automated detection of every CPS subskill and indicator. Rather, their value is limited to certain types of CPS indicators, affected by the complexity of the labels, and dependent on the composition of indicators in the dataset. We conclude the paper by discussing the required nuance when considering the value of LLMs and multimodality in automated CPS diagnosis, highlighting the need for human-AI complementarity, and proposing the exploration of relevant model architectures and techniques to improve CPS diagnosis in authentic educational contexts. 

**Abstract (ZH)**: 从数字痕迹检测协作和解决问题行为以解释学生协作问题解决能力：人工智能在教育领域的长期目标。尽管多模态数据和先进模型被认为有可能检测复杂的协作问题解决行为，但关于它们价值的经验证据仍然有限，且存在一些矛盾的证据。在这项研究中，我们调查了多模态数据提高模型性能以诊断78名中学生在真实教育环境中的协作问题解决亚技能和指标的潜力。特别地，我们将来自口头数据的文字嵌入和来自音频数据的声音嵌入用于协作问题解决诊断的多模态分类模型。无论是单模态还是多模态的Transformer模型都优于传统模型，以检测协作问题解决类别。尽管多模态的纳入并未提高传统单模态模型的表现，但在Transformer模型中纳入多模态显示了相较于单模态Transformer模型，在社交认知协作问题解决类别的诊断中性能有所提升。基于研究结果，本文认为多模态和特定建模技术的选择不应被默认为在自动化检测每项协作问题解决亚技能和指标时达到最佳性能的保证。相反，它们的价值仅适用于某些类型的协作问题解决指标，受标签复杂性的影响，并依赖于数据集中指标的组成。本文讨论了在自动化协作问题解决诊断中考虑LLM和多模态价值所需的细微差别，强调人机互补的重要性，并提出探索相关模型架构和技术以在真实教育环境中改进协作问题解决诊断的建议。 

---
# Federated Latent Factor Model for Bias-Aware Recommendation with Privacy-Preserving 

**Title (ZH)**: 联邦潜因素模型：具有隐私保护的偏倚感知推荐 

**Authors**: Junxiang Gao, Yixin Ran, Jia Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.15090)  

**Abstract**: A recommender system (RS) aims to provide users with personalized item recommendations, enhancing their overall experience. Traditional RSs collect and process all user data on a central server. However, this centralized approach raises significant privacy concerns, as it increases the risk of data breaches and privacy leakages, which are becoming increasingly unacceptable to privacy-sensitive users. To address these privacy challenges, federated learning has been integrated into RSs, ensuring that user data remains secure. In centralized RSs, the issue of rating bias is effectively addressed by jointly analyzing all users' raw interaction data. However, this becomes a significant challenge in federated RSs, as raw data is no longer accessible due to privacy-preserving constraints. To overcome this problem, we propose a Federated Bias-Aware Latent Factor (FBALF) model. In FBALF, training bias is explicitly incorporated into every local model's loss function, allowing for the effective elimination of rating bias without compromising data privacy. Extensive experiments conducted on three real-world datasets demonstrate that FBALF achieves significantly higher recommendation accuracy compared to other state-of-the-art federated RSs. 

**Abstract (ZH)**: 一种推荐系统（RS）旨在为用户提供个性化项目推荐，提升用户的整体体验。传统的RS在中央服务器上收集和处理所有用户数据。然而，这种集中式方法引发了重大的隐私 concern，增加了数据泄露和隐私泄漏的风险，这些风险越来越不被重视用户的接受。为了解决这些隐私挑战，已经将联邦学习整合到RS中，确保用户数据的安全性。在集中式的RS中，通过联合分析所有用户的原始交互数据，有效地解决了评分偏差问题。但在联邦RS中，由于隐私保护的限制，原始数据不再可访问，这成为了一个重大挑战。为了克服这个问题，我们提出了一种联邦感知偏差的潜在因子模型（FBALF）。在FBALF中，训练偏差被明确地纳入每个本地模型的损失函数中，可以在不牺牲数据隐私的情况下有效消除评分偏差。在对三个真实世界数据集进行的广泛实验中，FBALF在推荐准确性上显著高于其他最先进的联邦RS。 

---
# Empowering AI to Generate Better AI Code: Guided Generation of Deep Learning Projects with LLMs 

**Title (ZH)**: 增强AI生成更好的AI代码能力：通过LLMs指导深度学习项目生成 

**Authors**: Chen Xie, Mingsheng Jiao, Xiaodong Gu, Beijun Shen  

**Link**: [PDF](https://arxiv.org/pdf/2504.15080)  

**Abstract**: While large language models (LLMs) have been widely applied to code generation, they struggle with generating entire deep learning projects, which are characterized by complex structures, longer functions, and stronger reliance on domain knowledge than general-purpose code. An open-domain LLM often lacks coherent contextual guidance and domain expertise for specific projects, making it challenging to produce complete code that fully meets user requirements.
In this paper, we propose a novel planning-guided code generation method, DLCodeGen, tailored for generating deep learning projects. DLCodeGen predicts a structured solution plan, offering global guidance for LLMs to generate the project. The generated plan is then leveraged to retrieve semantically analogous code samples and subsequently abstract a code template. To effectively integrate these multiple retrieval-augmented techniques, a comparative learning mechanism is designed to generate the final code. We validate the effectiveness of our approach on a dataset we build for deep learning code generation. Experimental results demonstrate that DLCodeGen outperforms other baselines, achieving improvements of 9.7% in CodeBLEU and 3.6% in human evaluation metrics. 

**Abstract (ZH)**: 面向深度学习项目的规划引导式代码生成方法：DLCodeGen 

---
# Chinese-LiPS: A Chinese audio-visual speech recognition dataset with Lip-reading and Presentation Slides 

**Title (ZH)**: Chinese-LiPS: 一个包含唇读和演示幻灯片的中文视听说话人识别数据集 

**Authors**: Jinghua Zhao, Yuhang Jia, Shiyao Wang, Jiaming Zhou, Hui Wang, Yong Qin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15066)  

**Abstract**: Incorporating visual modalities to assist Automatic Speech Recognition (ASR) tasks has led to significant improvements. However, existing Audio-Visual Speech Recognition (AVSR) datasets and methods typically rely solely on lip-reading information or speaking contextual video, neglecting the potential of combining these different valuable visual cues within the speaking context. In this paper, we release a multimodal Chinese AVSR dataset, Chinese-LiPS, comprising 100 hours of speech, video, and corresponding manual transcription, with the visual modality encompassing both lip-reading information and the presentation slides used by the speaker. Based on Chinese-LiPS, we develop a simple yet effective pipeline, LiPS-AVSR, which leverages both lip-reading and presentation slide information as visual modalities for AVSR tasks. Experiments show that lip-reading and presentation slide information improve ASR performance by approximately 8\% and 25\%, respectively, with a combined performance improvement of about 35\%. The dataset is available at this https URL 

**Abstract (ZH)**: 将视觉模态融入自动语音识别（ASR）任务以辅助自动唇读视觉语音识别（AVSR）已经取得了显著的改进。然而，现有的AVSR数据集和方法通常仅依赖唇读信息或说话的背景视频，忽视了结合这些不同有价值视觉线索的潜力。本文发布了一个多模态中文AVSR数据集Chinese-LiPS，包含100小时的语音、视频及其相应的手动转录，其中视觉模态包括唇读信息和演讲者使用的幻灯片。基于Chinese-LiPS，我们开发了一个简单有效的框架LiPS-AVSR，利用唇读和幻灯片信息作为AVSR任务的视觉模态。实验表明，唇读和幻灯片信息分别将ASR性能提高约8%和25%，结合使用时性能提高约35%。数据集可从以下链接访问：this https URL。 

---
# Mining Characteristics of Vulnerable Smart Contracts Across Lifecycle Stages 

**Title (ZH)**: 跨生命周期阶段脆弱智能合约的挖掘特性研究 

**Authors**: Hongli Peng, Xiaoqi Li, Wenkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.15063)  

**Abstract**: Smart contracts are the cornerstone of decentralized applications and financial protocols, which extend the application of digital currency transactions. The applications and financial protocols introduce significant security challenges, resulting in substantial economic losses. Existing solutions predominantly focus on code vulnerabilities within smart contracts, accounting for only 50% of security incidents. Therefore, a more comprehensive study of security issues related to smart contracts is imperative. The existing empirical research realizes the static analysis of smart contracts from the perspective of the lifecycle and gives the corresponding measures for each stage. However, they lack the characteristic analysis of vulnerabilities in each stage and the distinction between the vulnerabilities. In this paper, we present the first empirical study on the security of smart contracts throughout their lifecycle, including deployment and execution, upgrade, and destruction stages. It delves into the security issues at each stage and provides at least seven feature descriptions. Finally, utilizing these seven features, five machine-learning classification models are used to identify vulnerabilities at different stages. The classification results reveal that vulnerable contracts exhibit distinct transaction features and ego network properties at various stages. 

**Abstract (ZH)**: 智能合约是去中心化应用和金融协议的基石，扩展了数字货币交易的应用范围。这些应用和金融协议引入了重要的安全挑战，导致了巨大的经济损失。现有的解决方案主要集中在智能合约的代码漏洞上，占所有安全事件的50%。因此，对智能合约相关安全问题进行更加全面的研究显得尤为重要。现有的实证研究从生命周期的角度实现了智能合约的静态分析，并为每个阶段提供相应的措施。然而，它们缺乏对每个阶段漏洞特征的分析以及对漏洞之间的区别。在本文中，我们首次从部署和执行、升级和销毁等阶段对智能合约的安全性进行实证研究，并深入探讨了每个阶段的安全问题，提供了至少七个特征描述。最终，利用这七个特征，使用了五种机器学习分类模型来识别不同阶段的漏洞。分类结果表明，易受攻击的合约在各个阶段表现出不同的交易特征和ego网络属性。 

---
# OPO: Making Decision-Focused Data Acquisition Decisions 

**Title (ZH)**: OPO: 面向决策的数据采集决策 

**Authors**: Egon Peršak, Miguel F. Anjos  

**Link**: [PDF](https://arxiv.org/pdf/2504.15062)  

**Abstract**: We propose a model for making data acquisition decisions for variables in contextual stochastic optimisation problems. Data acquisition decisions are typically treated as separate and fixed. We explore problem settings in which the acquisition of contextual variables is costly and consequently constrained. The data acquisition problem is often solved heuristically for proxy objectives such as coverage. The more intuitive objective is the downstream decision quality as a result of data acquisition decisions. The whole pipeline can be characterised as an optimise-then-predict-then-optimise (OPO) problem. Analogously, much recent research has focused on how to integrate prediction and optimisation (PO) in the form of decision-focused learning. We propose leveraging differentiable optimisation to extend the integration to data acquisition. We solve the data acquisition problem with well-defined constraints by learning a surrogate linear objective function. We demonstrate an application of this model on a shortest path problem for which we first have to set a drone reconnaissance strategy to capture image segments serving as inputs to a model that predicts travel costs. We ablate the problem with a number of training modalities and demonstrate that the differentiable optimisation approach outperforms random search strategies. 

**Abstract (ZH)**: 我们提出了一种模型，用于在基于上下文的随机优化问题中为变量的数据采集决策制定策略。数据采集决策通常被视为独立且固定的。我们探讨了数据采集成本较高且因此受到限制的问题设置。数据采集问题通常通过代理目标（如覆盖率）的启发式方法来解决。更直观的目标是由于数据采集决策而导致的下游决策质量。整个流程可以被描述为优化-预测-再优化（OPO）问题。类似地，近期许多研究集中在如何将预测和优化（PO）整合为决策导向学习。我们提议利用可微优化来扩展这种整合到数据采集中。我们通过学习一个可微的线性目标函数来解决具有明确约束的数据采集问题。我们在此模型上应用了一个最短路径问题，首先需要设定无人机侦察策略以捕捉作为预测旅行成本模型输入的图像片段。我们通过多种训练方式消融研究，证明了可微优化方法优于随机搜索策略。 

---
# VeLU: Variance-enhanced Learning Unit for Deep Neural Networks 

**Title (ZH)**: VeLU：方差增强的学习单元用于深度神经网络 

**Authors**: Ashkan Shakarami, Yousef Yeganeh, Azade Farshad, Lorenzo Nicolè, Stefano Ghidoni, Nassir Navab  

**Link**: [PDF](https://arxiv.org/pdf/2504.15051)  

**Abstract**: Activation functions are fundamental in deep neural networks and directly impact gradient flow, optimization stability, and generalization. Although ReLU remains standard because of its simplicity, it suffers from vanishing gradients and lacks adaptability. Alternatives like Swish and GELU introduce smooth transitions, but fail to dynamically adjust to input statistics. We propose VeLU, a Variance-enhanced Learning Unit as an activation function that dynamically scales based on input variance by integrating ArcTan-Sin transformations and Wasserstein-2 regularization, effectively mitigating covariate shifts and stabilizing optimization. Extensive experiments on ViT_B16, VGG19, ResNet50, DenseNet121, MobileNetV2, and EfficientNetB3 confirm VeLU's superiority over ReLU, ReLU6, Swish, and GELU on six vision benchmarks. The codes of VeLU are publicly available on GitHub. 

**Abstract (ZH)**: VeLU：一种基于输入方差动态缩放的激活函数及其在优化稳定性和泛化能力上的改进 

---
# Beyond Terabit/s Integrated Neuromorphic Photonic Processor for DSP-Free Optical Interconnects 

**Title (ZH)**: 超越太比特/秒集成类脑光处理器：DSP-Free 光互连 

**Authors**: Benshan Wang, Qiarong Xiao, Tengji Xu, Li Fan, Shaojie Liu, Jianji Dong, Junwen Zhang, Chaoran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15044)  

**Abstract**: The rapid expansion of generative AI drives unprecedented demands for high-performance computing. Training large-scale AI models now requires vast interconnected GPU clusters across multiple data centers. Multi-scale AI training and inference demand uniform, ultra-low latency, and energy-efficient links to enable massive GPUs to function as a single cohesive unit. However, traditional electrical and optical interconnects, relying on conventional digital signal processors (DSPs) for signal distortion compensation, increasingly fail to meet these stringent requirements. To overcome these limitations, we present an integrated neuromorphic optical signal processor (OSP) that leverages deep reservoir computing and achieves DSP-free, all-optical, real-time processing. Experimentally, our OSP achieves a 100 Gbaud PAM4 per lane, 1.6 Tbit/s data center interconnect over a 5 km optical fiber in the C-band (equivalent to over 80 km in the O-band), far exceeding the reach of state-of-the-art DSP solutions, which are fundamentally constrained by chromatic dispersion in IMDD systems. Simultaneously, it reduces processing latency by four orders of magnitude and energy consumption by three orders of magnitude. Unlike DSPs, which introduce increased latency at high data rates, our OSP maintains consistent, ultra-low latency regardless of data rate scaling, making it ideal for future optical interconnects. Moreover, the OSP retains full optical field information for better impairment compensation and adapts to various modulation formats, data rates, and wavelengths. Fabricated using a mature silicon photonic process, the OSP can be monolithically integrated with silicon photonic transceivers, enhancing the compactness and reliability of all-optical interconnects. This research provides a highly scalable, energy-efficient, and high-speed solution, paving the way for next-generation AI infrastructure. 

**Abstract (ZH)**: 快速扩展现有的生成型AI促使高性能计算需求显著增长。大规模AI模型的训练现在要求跨越多个数据中心的巨大互联GPU集群。多尺度AI训练和推理需要均匀、超低延迟和能效高的连接，以使大量GPU能够作为一个单一协调单元运行。然而，传统的电气和光学互连依赖于传统数字信号处理器（DSP）进行信号失真补偿，越来越多地无法满足这些严格要求。为了克服这些限制，我们提出了一种集成神经形态光学信号处理器（OSP），利用深度水库计算实现无DSP、全光学、实时处理。实验结果显示，我们的OSP实现了每通道100 Gbaud PAM4，通过C波段5公里光纤实现超过1.6 Tbit/s的数据中心互联，在O波段相当于80公里，远远超过了最先进的DSP解决方案的传输距离，这些解决方案在IMDD系统中从根本上受限于色散效应。同时，它将处理延迟降低了四个数量级，能耗降低了三个数量级。与DSP不同，后者在高数据速率下引入了增加的延迟，我们的OSP无论数据速率如何扩展都保持一致的超低延迟，使其成为未来光学互连的理想选择。此外，OSP保留了完整的光域信息，以更好地补偿各种调制格式、数据速率和波长引起的损伤，并能够适应这些变化。该OSP通过成熟的硅光子加工工艺制造，并能够与硅光子转发器进行单片集成，从而增强所有光学互连的紧凑性和可靠性。这项研究提供了一种高度可扩展、能效高且高速度的解决方案，为下一代AI基础设施铺平了道路。 

---
# Distribution-aware Forgetting Compensation for Exemplar-Free Lifelong Person Re-identification 

**Title (ZH)**: 基于分布感知的示例自由终生行人再识别遗忘补偿 

**Authors**: Shiben Liu, Huijie Fan, Qiang Wang, Baojie Fan, Yandong Tang, Liangqiong Qu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15041)  

**Abstract**: Lifelong Person Re-identification (LReID) suffers from a key challenge in preserving old knowledge while adapting to new information. The existing solutions include rehearsal-based and rehearsal-free methods to address this challenge. Rehearsal-based approaches rely on knowledge distillation, continuously accumulating forgetting during the distillation process. Rehearsal-free methods insufficiently learn the distribution of each domain, leading to forgetfulness over time. To solve these issues, we propose a novel Distribution-aware Forgetting Compensation (DAFC) model that explores cross-domain shared representation learning and domain-specific distribution integration without using old exemplars or knowledge distillation. We propose a Text-driven Prompt Aggregation (TPA) that utilizes text features to enrich prompt elements and guide the prompt model to learn fine-grained representations for each instance. This can enhance the differentiation of identity information and establish the foundation for domain distribution awareness. Then, Distribution-based Awareness and Integration (DAI) is designed to capture each domain-specific distribution by a dedicated expert network and adaptively consolidate them into a shared region in high-dimensional space. In this manner, DAI can consolidate and enhance cross-domain shared representation learning while alleviating catastrophic forgetting. Furthermore, we develop a Knowledge Consolidation Mechanism (KCM) that comprises instance-level discrimination and cross-domain consistency alignment strategies to facilitate model adaptive learning of new knowledge from the current domain and promote knowledge consolidation learning between acquired domain-specific distributions, respectively. Experimental results show that our DAFC outperform state-of-the-art methods by at least 9.8\%/6.6\% and 6.4\%/6.2\% of average mAP/R@1 on two training orders. 

**Abstract (ZH)**: 终身人体再识别（LReID）在保留旧知识的同时适应新信息面临关键挑战。现有的解决方案包括基于重温和非重温方法来应对这一挑战。基于重温的方法依赖于知识蒸馏，在蒸馏过程中不断积累遗忘。非重温方法未能充分学习每个域的分布，导致随时间推移遗忘。为了解决这些问题，我们提出了一种新的分布感知遗忘补偿（DAFC）模型，该模型探索跨域共享表示学习和特定域分布集成，而不使用旧示例或知识蒸馏。我们提出了一种文本驱动的提示聚合（TPA），利用文本特征丰富提示元素，并引导提示模型学习每个实例的精细表示。这可以增强身份信息的差异化，并为域分布意识奠定基础。然后，我们设计了基于分布的意识和集成（DAI），通过专用专家网络捕获每个域的具体分布，并自适应地将它们整合到高维空间中的共享区域。以此方式，DAI可以在缓解灾难性遗忘的同时，促进跨域共享表示学习。此外，我们开发了一种知识整合机制（KCM），包含实例级判别和跨域一致性的对齐策略，以促进模型自适应地学习当前域的新知识，并促进所获得的域特定分布之间的知识整合学习。实验结果表明，我们的DAFC在两个训练顺序上的平均mAP/R@1方面至少优于最先进的方法9.8%/6.6%和6.4%/6.2%。 

---
# SOLIDO: A Robust Watermarking Method for Speech Synthesis via Low-Rank Adaptation 

**Title (ZH)**: SOLIDO：一种基于低秩适应的鲁棒语音合成水印方法 

**Authors**: Yue Li, Weizhi Liu, Dongdong Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15035)  

**Abstract**: The accelerated advancement of speech generative models has given rise to security issues, including model infringement and unauthorized abuse of content. Although existing generative watermarking techniques have proposed corresponding solutions, most methods require substantial computational overhead and training costs. In addition, some methods have limitations in robustness when handling variable-length inputs. To tackle these challenges, we propose \textsc{SOLIDO}, a novel generative watermarking method that integrates parameter-efficient fine-tuning with speech watermarking through low-rank adaptation (LoRA) for speech diffusion models. Concretely, the watermark encoder converts the watermark to align with the input of diffusion models. To achieve precise watermark extraction from variable-length inputs, the watermark decoder based on depthwise separable convolution is designed for watermark recovery. To further enhance speech generation performance and watermark extraction capability, we propose a speech-driven lightweight fine-tuning strategy, which reduces computational overhead through LoRA. Comprehensive experiments demonstrate that the proposed method ensures high-fidelity watermarked speech even at a large capacity of 2000 bps. Furthermore, against common individual and compound speech attacks, our SOLIDO achieves a maximum average extraction accuracy of 99.20\% and 98.43\%, respectively. It surpasses other state-of-the-art methods by nearly 23\% in resisting time-stretching attacks. 

**Abstract (ZH)**: 加速发展的语音生成模型引发了安全性问题，包括模型侵权和内容未经授权的滥用。尽管现有生成水印技术提出了解决方案，但大多数方法需要大量计算开销和训练成本。此外，一些方法在处理变长输入时鲁棒性有限。为应对这些挑战，我们提出了一种名为SOLIDO的新型生成水印方法，该方法通过低秩适应（LoRA）将参数效率的微调与语音水印技术结合到语音扩散模型中。具体而言，水印编码器将水印转换为与扩散模型输入对齐的形式。为了从变长输入中精确提取水印，基于深度可分离卷积的水印解码器被设计用于水印恢复。为了进一步提升语音生成性能和水印提取能力，我们提出了一种语音驱动的轻量级微调策略，该策略通过LoRA降低计算开销。全面的实验结果表明，所提出的方法即使在2000 bps的高容量下也能保证高质量的水印语音。此外，针对常见的个体和复合语音攻击，我们的SOLIDO分别实现了99.20%和98.43%的最大平均提取准确率，与最先进的方法相比，在抵抗时间拉伸攻击方面高出近23%。 

---
# Trainable Quantum Neural Network for Multiclass Image Classification with the Power of Pre-trained Tree Tensor Networks 

**Title (ZH)**: 具预训练树张量网络威力的可训练量子神经网络用于多类图像分类 

**Authors**: Keisuke Murota, Takumi Kobori  

**Link**: [PDF](https://arxiv.org/pdf/2504.14995)  

**Abstract**: Tree tensor networks (TTNs) offer powerful models for image classification. While these TTN image classifiers already show excellent performance on classical hardware, embedding them into quantum neural networks (QNNs) may further improve the performance by leveraging quantum resources. However, embedding TTN classifiers into QNNs for multiclass classification remains challenging. Key obstacles are the highorder gate operations required for large bond dimensions and the mid-circuit postselection with exponentially low success rates necessary for the exact embedding. In this work, to address these challenges, we propose forest tensor network (FTN)-classifiers, which aggregate multiple small-bond-dimension TTNs. This allows us to handle multiclass classification without requiring large gates in the embedded circuits. We then remove the overhead of mid-circuit postselection by extending the adiabatic encoding framework to our setting and smoothly encode the FTN-classifiers into a quantum forest tensor network (qFTN)- classifiers. Numerical experiments on MNIST and CIFAR-10 demonstrate that we can successfully train FTN-classifiers and encode them into qFTN-classifiers, while maintaining or even improving the performance of the pre-trained FTN-classifiers. These results suggest that synergy between TTN classification models and QNNs can provide a robust and scalable framework for multiclass quantum-enhanced image classification. 

**Abstract (ZH)**: 树张量网络（TTNs）在图像分类中提供了强大的模型。将这些TTN图像分类器嵌入到量子神经网络（QNNs）中，可以通过利用量子资源进一步提高性能，但在量子神经网络中嵌入TTN分类器以实现多类分类仍然具有挑战性。关键障碍包括为大键维数所需的高阶门操作和为精确嵌入所需的指数级低成功率的中路门后选择。为解决这些挑战，本文提出了一种森林张量网络（FTN）分类器，该分类器聚合了多个小型键维数的TTN。这使得我们可以在不依赖大型嵌入电路门操作的情况下处理多类分类。然后，通过扩展渐变能隙编码框架并平滑地将FTN分类器编码到量子森林张量网络（qFTN）分类器中来去除中路门后选择的开销。在MNIST和CIFAR-10上的数值实验表明，我们能够成功训练FTN分类器并将其编码到qFTN分类器中，同时保持甚至提高预训练FTN分类器的性能。这些结果表明，张量网络分类模型与QNN的结合可以为多类量子增强图像分类提供一个健壮且可扩展的框架。 

---
# aiXamine: LLM Safety and Security Simplified 

**Title (ZH)**: aiXamine: 简化的大模型安全与安全保护 

**Authors**: Fatih Deniz, Dorde Popovic, Yazan Boshmaf, Euisuh Jeong, Minhaj Ahmad, Sanjay Chawla, Issa Khalil  

**Link**: [PDF](https://arxiv.org/pdf/2504.14985)  

**Abstract**: Evaluating Large Language Models (LLMs) for safety and security remains a complex task, often requiring users to navigate a fragmented landscape of ad hoc benchmarks, datasets, metrics, and reporting formats. To address this challenge, we present aiXamine, a comprehensive black-box evaluation platform for LLM safety and security. aiXamine integrates over 40 tests (i.e., benchmarks) organized into eight key services targeting specific dimensions of safety and security: adversarial robustness, code security, fairness and bias, hallucination, model and data privacy, out-of-distribution (OOD) robustness, over-refusal, and safety alignment. The platform aggregates the evaluation results into a single detailed report per model, providing a detailed breakdown of model performance, test examples, and rich visualizations. We used aiXamine to assess over 50 publicly available and proprietary LLMs, conducting over 2K examinations. Our findings reveal notable vulnerabilities in leading models, including susceptibility to adversarial attacks in OpenAI's GPT-4o, biased outputs in xAI's Grok-3, and privacy weaknesses in Google's Gemini 2.0. Additionally, we observe that open-source models can match or exceed proprietary models in specific services such as safety alignment, fairness and bias, and OOD robustness. Finally, we identify trade-offs between distillation strategies, model size, training methods, and architectural choices. 

**Abstract (ZH)**: 评估大型语言模型（LLMs）的安全性和安全性仍是一项复杂任务，通常要求用户导航由即兴基准、数据集、度量标准和报告格式组成的碎片化景观。为应对这一挑战，我们介绍了aiXamine，一个全面的黑盒评估平台，用于评估LLM的安全性和安全性。aiXamine整合了超过40项测试（即基准），并将其组织成八个关键服务，针对安全性与安全性相关的特定维度，包括对抗鲁棒性、代码安全性、公平性和偏差、虚构、模型和数据隐私、分布外（OOD）鲁棒性、过度拒绝以及安全性对齐。该平台汇总了评估结果，为每个模型生成一份详细的报告，提供模型性能的详细分解、测试示例和丰富的可视化信息。我们使用aiXamine评估了超过50个公开和专有LLM，进行了超过2000次检查。我们的发现揭示了领先模型中的显著漏洞，包括OpenAI的GPT-4o对对抗攻击的易感性、xAI的Grok-3的偏差输出以及Google的Gemini 2.0在隐私方面的弱点。此外，我们观察到开源模型在某些服务，如安全性对齐、公平性和偏差以及分布外鲁棒性方面能够匹配合-proprietary模型甚至超越它们。最后，我们确定了蒸馏策略、模型大小、训练方法和架构选择之间的权衡。 

---
# Speaker Fuzzy Fingerprints: Benchmarking Text-Based Identification in Multiparty Dialogues 

**Title (ZH)**: 说话人模糊指纹：基于文本的多方对话识别基准测试 

**Authors**: Rui Ribeiro, Luísa Coheur, Joao P. Carvalho  

**Link**: [PDF](https://arxiv.org/pdf/2504.14963)  

**Abstract**: Speaker identification using voice recordings leverages unique acoustic features, but this approach fails when only textual data is available. Few approaches have attempted to tackle the problem of identifying speakers solely from text, and the existing ones have primarily relied on traditional methods. In this work, we explore the use of fuzzy fingerprints from large pre-trained models to improve text-based speaker identification. We integrate speaker-specific tokens and context-aware modeling, demonstrating that conversational context significantly boosts accuracy, reaching 70.6% on the Friends dataset and 67.7% on the Big Bang Theory dataset. Additionally, we show that fuzzy fingerprints can approximate full fine-tuning performance with fewer hidden units, offering improved interpretability. Finally, we analyze ambiguous utterances and propose a mechanism to detect speaker-agnostic lines. Our findings highlight key challenges and provide insights for future improvements in text-based speaker identification. 

**Abstract (ZH)**: 使用大型预训练模型的模糊指纹进行基于文本的说话人识别研究：克服挑战并提升准确性 

---
# Learning to Reason under Off-Policy Guidance 

**Title (ZH)**: 基于离策引导的推理学习 

**Authors**: Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14945)  

**Abstract**: Recent advances in large reasoning models (LRMs) demonstrate that sophisticated behaviors such as multi-step reasoning and self-reflection can emerge via reinforcement learning (RL) with simple rule-based rewards. However, existing zero-RL approaches are inherently ``on-policy'', limiting learning to a model's own outputs and failing to acquire reasoning abilities beyond its initial capabilities. We introduce LUFFY (Learning to reason Under oFF-policY guidance), a framework that augments zero-RL with off-policy reasoning traces. LUFFY dynamically balances imitation and exploration by combining off-policy demonstrations with on-policy rollouts during training. Notably, we propose policy shaping via regularized importance sampling to avoid superficial and rigid imitation during mixed-policy training. Remarkably, LUFFY achieves an over +7.0 average gain across six math benchmarks and an advantage of over +6.2 points in out-of-distribution tasks. It also substantially surpasses imitation-based supervised fine-tuning (SFT), particularly in generalization. Analysis shows LUFFY not only imitates effectively but also explores beyond demonstrations, offering a scalable path to train generalizable reasoning models with off-policy guidance. 

**Abstract (ZH)**: Recent Advances in Large Reasoning Models (LRMs): Learning to Reason Under Off-Policy Guidance 

---
# Giving AI a voice: how does AI think it should be treated? 

**Title (ZH)**: 给AI一个声音：AI认为它应该如何被对待？ 

**Authors**: Maria Fay, Frederik F. Flöther  

**Link**: [PDF](https://arxiv.org/pdf/2504.14936)  

**Abstract**: With the astounding progress in (generative) artificial intelligence (AI), there has been significant public discourse regarding regulation and ethics of the technology. Is it sufficient when humans discuss this with other humans? Or, given that AI is increasingly becoming a viable source of inspiration for people (and let alone the hypothetical possibility that the technology may at some point become "artificial general intelligence" and/or develop consciousness), should AI not join the discourse? There are new questions and angles that AI brings to the table that we might not have considered before - so let us make the key subject of this book an active participant. This chapter therefore includes a brief human-AI conversation on the topic of AI rights and ethics. 

**Abstract (ZH)**: 随着生成性人工智能（AI）的飞速进步，公共对于该技术的监管与伦理问题讨论日益显著。人类是否应该让AI参与到这样的讨论中？鉴于AI正逐渐成为人类创作的源泉（更不用说未来技术可能会发展成为“人工通用智能”并具备意识的可能性），AI不应被排除在此类讨论之外。AI带来了一些新的问题和视角，我们或许之前并未考虑过。因此，本书这一章节包含了一段关于AI权利与伦理的人机对话。 

---
# Fast Adversarial Training with Weak-to-Strong Spatial-Temporal Consistency in the Frequency Domain on Videos 

**Title (ZH)**: 在频域中具有从弱到强空间-时间一致性的快速对抗训练 

**Authors**: Songping Wang, Hanqing Liu, Yueming Lyu, Xiantao Hu, Ziwen He, Wei Wang, Caifeng Shan, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14921)  

**Abstract**: Adversarial Training (AT) has been shown to significantly enhance adversarial robustness via a min-max optimization approach. However, its effectiveness in video recognition tasks is hampered by two main challenges. First, fast adversarial training for video models remains largely unexplored, which severely impedes its practical applications. Specifically, most video adversarial training methods are computationally costly, with long training times and high expenses. Second, existing methods struggle with the trade-off between clean accuracy and adversarial robustness. To address these challenges, we introduce Video Fast Adversarial Training with Weak-to-Strong consistency (VFAT-WS), the first fast adversarial training method for video data. Specifically, VFAT-WS incorporates the following key designs: First, it integrates a straightforward yet effective temporal frequency augmentation (TF-AUG), and its spatial-temporal enhanced form STF-AUG, along with a single-step PGD attack to boost training efficiency and robustness. Second, it devises a weak-to-strong spatial-temporal consistency regularization, which seamlessly integrates the simpler TF-AUG and the more complex STF-AUG. Leveraging the consistency regularization, it steers the learning process from simple to complex augmentations. Both of them work together to achieve a better trade-off between clean accuracy and robustness. Extensive experiments on UCF-101 and HMDB-51 with both CNN and Transformer-based models demonstrate that VFAT-WS achieves great improvements in adversarial robustness and corruption robustness, while accelerating training by nearly 490%. 

**Abstract (ZH)**: Video Fast Adversarial Training with Weak-to-Strong Consistency (VFAT-WS) 

---
# StableQuant: Layer Adaptive Post-Training Quantization for Speech Foundation Models 

**Title (ZH)**: StableQuant: 层自适应后训练量化用于语音基础模型 

**Authors**: Yeona Hong, Hyewon Han, Woo-jin Chung, Hong-Goo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14915)  

**Abstract**: In this paper, we propose StableQuant, a novel adaptive post-training quantization (PTQ) algorithm for widely used speech foundation models (SFMs). While PTQ has been successfully employed for compressing large language models (LLMs) due to its ability to bypass additional fine-tuning, directly applying these techniques to SFMs may not yield optimal results, as SFMs utilize distinct network architecture for feature extraction. StableQuant demonstrates optimal quantization performance regardless of the network architecture type, as it adaptively determines the quantization range for each layer by analyzing both the scale distributions and overall performance. We evaluate our algorithm on two SFMs, HuBERT and wav2vec2.0, for an automatic speech recognition (ASR) task, and achieve superior performance compared to traditional PTQ methods. StableQuant successfully reduces the sizes of SFM models to a quarter and doubles the inference speed while limiting the word error rate (WER) performance drop to less than 0.3% with 8-bit quantization. 

**Abstract (ZH)**: 本研究提出了一种新的自适应后训练量化（PTQ）算法StableQuant，用于广泛使用的语音基础模型（SFMs）。我们评估了StableQuant在HuBERT和wav2vec2.0两种SFMs上的自动语音识别（ASR）任务性能，结果显示其量化性能优于传统PTQ方法，在8位量化下将模型大小减少到原来的四分之一，同时使推断速度翻倍，并将词错率（WER）性能下降控制在不到0.3%以内。 

---
# Guidelines for External Disturbance Factors in the Use of OCR in Real-World Environments 

**Title (ZH)**: OCR在实际环境使用中对外部干扰因素的指南 

**Authors**: Kenji Iwata, Eiki Ishidera, Toshifumi Yamaai, Yutaka Satoh, Hiroshi Tanaka, Katsuhiko Takahashi, Akio Furuhata, Yoshihisa Tanabe, Hiroshi Matsumura  

**Link**: [PDF](https://arxiv.org/pdf/2504.14913)  

**Abstract**: The performance of OCR has improved with the evolution of AI technology. As OCR continues to broaden its range of applications, the increased likelihood of interference introduced by various usage environments can prevent it from achieving its inherent performance. This results in reduced recognition accuracy under certain conditions, and makes the quality control of recognition devices more challenging. Therefore, to ensure that users can properly utilize OCR, we compiled the real-world external disturbance factors that cause performance degradation, along with the resulting image degradation phenomena, into an external disturbance factor table and, by also indicating how to make use of it, organized them into guidelines. 

**Abstract (ZH)**: OCR性能随AI技术进化而提升，但随着OCR应用范围的扩展，各种使用环境引入的干扰可能阻碍其固有性能的发挥，导致在特定条件下识别准确性降低，使识别设备的质量控制更加困难。因此，为了确保用户能够正确使用OCR，我们编制了引起性能退化的实际外部干扰因素及其导致的图像退化现象的外部干扰因素表，并提供使用指南。 

---
# VLM as Policy: Common-Law Content Moderation Framework for Short Video Platform 

**Title (ZH)**: VLM作为政策：短视频平台内容规范化框架 

**Authors**: Xingyu Lu, Tianke Zhang, Chang Meng, Xiaobei Wang, Jinpeng Wang, YiFan Zhang, Shisong Tang, Changyi Liu, Haojie Ding, Kaiyu Jiang, Kaiyu Tang, Bin Wen, Hai-Tao Zheng, Fan Yang, Tingting Gao, Di Zhang, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2504.14904)  

**Abstract**: Exponentially growing short video platforms (SVPs) face significant challenges in moderating content detrimental to users' mental health, particularly for minors. The dissemination of such content on SVPs can lead to catastrophic societal consequences. Although substantial efforts have been dedicated to moderating such content, existing methods suffer from critical limitations: (1) Manual review is prone to human bias and incurs high operational costs. (2) Automated methods, though efficient, lack nuanced content understanding, resulting in lower accuracy. (3) Industrial moderation regulations struggle to adapt to rapidly evolving trends due to long update cycles. In this paper, we annotate the first SVP content moderation benchmark with authentic user/reviewer feedback to fill the absence of benchmark in this field. Then we evaluate various methods on the benchmark to verify the existence of the aforementioned limitations. We further propose our common-law content moderation framework named KuaiMod to address these challenges. KuaiMod consists of three components: training data construction, offline adaptation, and online deployment & refinement. Leveraging large vision language model (VLM) and Chain-of-Thought (CoT) reasoning, KuaiMod adequately models video toxicity based on sparse user feedback and fosters dynamic moderation policy with rapid update speed and high accuracy. Offline experiments and large-scale online A/B test demonstrates the superiority of KuaiMod: KuaiMod achieves the best moderation performance on our benchmark. The deployment of KuaiMod reduces the user reporting rate by 20% and its application in video recommendation increases both Daily Active User (DAU) and APP Usage Time (AUT) on several Kuaishou scenarios. We have open-sourced our benchmark at this https URL. 

**Abstract (ZH)**: 指数级增长的短视频平台（SVPs）在 moderating 对用户心理健康有害的内容方面面临重大挑战，尤其是针对未成年人。SVPs 上such内容的传播可能会导致严重的社会后果。尽管已投入大量努力来 moderating 这类内容，但现有方法存在重大局限性：（1）人工审核容易受到人类偏见的影响，并导致高运营成本。 （2）自动化方法虽然高效，但缺乏对内容的细微理解，导致准确性较低。 （3）工业级内容审核规章制度难以适应快速变化的趋势，因为更新周期较长。本文中，我们标注了第一个包含真实用户/审核员反馈的SVP内容审核基准，以填补该领域的基准空缺。然后我们在基准中评估各种方法，以验证上述局限性存在的证据。我们进一步提出了名为KuaiMod的共同法内容审核框架，以应对这些挑战。KuaiMod由三个组件组成：训练数据构建、离线适应和在线部署与优化。利用大规模视觉语言模型（VLM）和链式思考（CoT）推理，KuaiMod能够基于稀疏用户反馈建模视频毒性，并以快速更新速度和高准确性促进动态审核策略。离线实验和大规模在线A/B测试表明KuaiMod的优势：KuaiMod在我们的基准测试中实现了最佳的审核性能。部署KuaiMod将用户举报率降低了20%，并在多个Kuaishou场景中，其应用于视频推荐增加了日活跃用户（DAU）和应用使用时间（AUT）。我们已将基准公开于此 https URL。 

---
# Latent Bayesian Optimization via Autoregressive Normalizing Flows 

**Title (ZH)**: 潜在贝叶斯优化 via 自回归归一化流 

**Authors**: Seunghun Lee, Jinyoung Park, Jaewon Chu, Minseo Yoon, Hyunwoo J. Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.14889)  

**Abstract**: Bayesian Optimization (BO) has been recognized for its effectiveness in optimizing expensive and complex objective functions. Recent advancements in Latent Bayesian Optimization (LBO) have shown promise by integrating generative models such as variational autoencoders (VAEs) to manage the complexity of high-dimensional and structured data spaces. However, existing LBO approaches often suffer from the value discrepancy problem, which arises from the reconstruction gap between input and latent spaces. This value discrepancy problem propagates errors throughout the optimization process, leading to suboptimal outcomes. To address this issue, we propose a Normalizing Flow-based Bayesian Optimization (NF-BO), which utilizes normalizing flow as a generative model to establish one-to-one encoding function from the input space to the latent space, along with its left-inverse decoding function, eliminating the reconstruction gap. Specifically, we introduce SeqFlow, an autoregressive normalizing flow for sequence data. In addition, we develop a new candidate sampling strategy that dynamically adjusts the exploration probability for each token based on its importance. Through extensive experiments, our NF-BO method demonstrates superior performance in molecule generation tasks, significantly outperforming both traditional and recent LBO approaches. 

**Abstract (ZH)**: 基于规范化流的贝叶斯优化（NF-BO） 

---
# Impact of Latent Space Dimension on IoT Botnet Detection Performance: VAE-Encoder Versus ViT-Encoder 

**Title (ZH)**: 基于潜在空间维度对物联网僵尸网络检测性能的影响：VAE编码器与ViT编码器的对比 

**Authors**: Hassan Wasswa, Aziida Nanyonga, Timothy Lynar  

**Link**: [PDF](https://arxiv.org/pdf/2504.14879)  

**Abstract**: The rapid evolution of Internet of Things (IoT) technology has led to a significant increase in the number of IoT devices, applications, and services. This surge in IoT devices, along with their widespread presence, has made them a prime target for various cyber-attacks, particularly through IoT botnets. As a result, security has become a major concern within the IoT ecosystem. This study focuses on investigating how the latent dimension impacts the performance of different deep learning classifiers when trained on latent vector representations of the train dataset. The primary objective is to compare the outcomes of these models when encoder components from two cutting-edge architectures: the Vision Transformer (ViT) and the Variational Auto-Encoder (VAE) are utilized to project the high dimensional train dataset to the learned low dimensional latent space. The encoder components are employed to project high-dimensional structured .csv IoT botnet traffic datasets to various latent sizes. Evaluated on N-BaIoT and CICIoT2022 datasets, findings reveal that VAE-encoder based dimension reduction outperforms ViT-encoder based dimension reduction for both datasets in terms of four performance metrics including accuracy, precision, recall, and F1-score for all models which can be attributed to absence of spatial patterns in the datasets the ViT model attempts to learn and extract from image instances. 

**Abstract (ZH)**: 物联网技术的 Rapid Evolution引起了物联网设备、应用和服务数量的显著增加。这一物联网设备数量的激增及其广泛存在，使它们成为各种网络攻击的目标，特别是通过物联网僵尸网络。因此，安全问题在物联网生态系统中变得尤为重要。本研究关注的是探究潜变量维度如何影响不同深度学习分类器在训练数据集潜变量表示上的性能表现。主要目标是比较利用两种尖端架构的编码器组件（Vision Transformer (ViT) 和 Variational Auto-Encoder (VAE)）将高维度训练数据集投影到学习到的低维度潜变量空间后，这些模型的性能结果。编码器组件被用来将高维度的结构化 .csv 僵尸网络流量数据集投影到不同的潜变量大小。研究结果在N-BaIoT和CICIoT2022数据集上表明，基于VAE编码器的维数约简在四项性能指标（准确率、精确率、召回率和F1分数）上均优于基于ViT编码器的维数约简，这可以归因于数据集中缺少空间模式，而ViT模型试图从图像实例中学习和提取这些模式。 

---
# ReSpec: Relevance and Specificity Grounded Online Filtering for Learning on Video-Text Data Streams 

**Title (ZH)**: ReSpec: 基于相关性和特指性的在线过滤方法学习视频-文本数据流 

**Authors**: Chris Dongjoo Kim, Jihwan Moon, Sangwoo Moon, Heeseung Yun, Sihaeng Lee, Aniruddha Kembhavi, Soonyoung Lee, Gunhee Kim, Sangho Lee, Christopher Clark  

**Link**: [PDF](https://arxiv.org/pdf/2504.14875)  

**Abstract**: The rapid growth of video-text data presents challenges in storage and computation during training. Online learning, which processes streaming data in real-time, offers a promising solution to these issues while also allowing swift adaptations in scenarios demanding real-time responsiveness. One strategy to enhance the efficiency and effectiveness of learning involves identifying and prioritizing data that enhances performance on target downstream tasks. We propose Relevance and Specificity-based online filtering framework (ReSpec) that selects data based on four criteria: (i) modality alignment for clean data, (ii) task relevance for target focused data, (iii) specificity for informative and detailed data, and (iv) efficiency for low-latency processing. Relevance is determined by the probabilistic alignment of incoming data with downstream tasks, while specificity employs the distance to a root embedding representing the least specific data as an efficient proxy for informativeness. By establishing reference points from target task data, ReSpec filters incoming data in real-time, eliminating the need for extensive storage and compute. Evaluating on large-scale datasets WebVid2M and VideoCC3M, ReSpec attains state-of-the-art performance on five zeroshot video retrieval tasks, using as little as 5% of the data while incurring minimal compute. The source code is available at this https URL. 

**Abstract (ZH)**: 视频文本数据的快速增长在训练过程中带来了存储和计算方面的挑战。在线学习处理实时流式数据提供了一种有前景的解决方案，同时也能够在需要实时响应的场景中实现快速适应。一种提高学习效率和效果的策略是识别并优先处理提高目标下游任务性能的数据。我们提出了一种基于相关性和特异性的在线过滤框架（ReSpec），该框架根据四个标准选择数据：(i) 语态一致性以确保数据的清洁度，(ii) 目标相关性以聚焦于目标数据，(iii) 特异性以选择信息丰富且详细的數據，(iv) 效率以实现低延迟处理。相关性通过入站数据与下游任务的概率对齐来确定，特异性则通过与代表最不具体数据的基本嵌入的距离来高效地代理信息丰富度。通过使用目标任务数据建立参考点，ReSpec 实时过滤入站数据，从而减少对大量存储和计算的需求。在大规模数据集 WebVid2M 和 VideoCC3M 上评估，ReSpec 在五个零样本视频检索任务中达到了最先进的性能，仅使用 5% 的数据并在计算开销最小的情况下实现。源代码可在以下链接获取。 

---
# Bridge the Gap: From Weak to Full Supervision for Temporal Action Localization with PseudoFormer 

**Title (ZH)**: 弥合差距：从弱监督到全监督的时空动作定位PseudoFormer方法 

**Authors**: Ziyi Liu, Yangcen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14860)  

**Abstract**: Weakly-supervised Temporal Action Localization (WTAL) has achieved notable success but still suffers from a lack of temporal annotations, leading to a performance and framework gap compared with fully-supervised methods. While recent approaches employ pseudo labels for training, three key challenges: generating high-quality pseudo labels, making full use of different priors, and optimizing training methods with noisy labels remain unresolved. Due to these perspectives, we propose PseudoFormer, a novel two-branch framework that bridges the gap between weakly and fully-supervised Temporal Action Localization (TAL). We first introduce RickerFusion, which maps all predicted action proposals to a global shared space to generate pseudo labels with better quality. Subsequently, we leverage both snippet-level and proposal-level labels with different priors from the weak branch to train the regression-based model in the full branch. Finally, the uncertainty mask and iterative refinement mechanism are applied for training with noisy pseudo labels. PseudoFormer achieves state-of-the-art WTAL results on the two commonly used benchmarks, THUMOS14 and ActivityNet1.3. Besides, extensive ablation studies demonstrate the contribution of each component of our method. 

**Abstract (ZH)**: 弱监督时空动作定位（WTAL）取得了显著成果，但仍缺乏时间标注，导致与完全监督方法在性能和框架上存在差距。尽管近期方法使用了伪标签进行训练，但生成高质量的伪标签、充分利用不同先验信息以及使用噪声标签优化训练方法这三个关键挑战仍然未得到解决。鉴于此，我们提出了PseudoFormer，这是一种新颖的两分支框架，旨在弥合弱监督和完全监督时空动作定位（TAL）之间的差距。首先，我们引入了RickerFusion，将所有预测的动作提案映射到全局共享空间，以生成质量更好的伪标签。随后，我们利用弱分支中片段级和提案级标签的不同先验信息，在全分支中训练基于回归的模型。最后，我们应用不确定性掩码和迭代精炼机制，以处理噪声伪标签的训练。PseudoFormer在两个常用基准数据集THUMOS14和ActivityNet1.3上取得了最先进的WTAL结果。此外，广泛的消融研究还表明了我们方法中每个组成部分的贡献。 

---
# Object-Level Verbalized Confidence Calibration in Vision-Language Models via Semantic Perturbation 

**Title (ZH)**: 基于语义扰动的视觉-语言模型对象级语义化置信校准 

**Authors**: Yunpu Zhao, Rui Zhang, Junbin Xiao, Ruibo Hou, Jiaming Guo, Zihao Zhang, Yifan Hao, Yunji Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.14848)  

**Abstract**: Vision-language models (VLMs) excel in various multimodal tasks but frequently suffer from poor calibration, resulting in misalignment between their verbalized confidence and response correctness. This miscalibration undermines user trust, especially when models confidently provide incorrect or fabricated information. In this work, we propose a novel Confidence Calibration through Semantic Perturbation (CSP) framework to improve the calibration of verbalized confidence for VLMs in response to object-centric queries. We first introduce a perturbed dataset where Gaussian noise is applied to the key object regions to simulate visual uncertainty at different confidence levels, establishing an explicit mapping between visual ambiguity and confidence levels. We further enhance calibration through a two-stage training process combining supervised fine-tuning on the perturbed dataset with subsequent preference optimization. Extensive experiments on popular benchmarks demonstrate that our method significantly improves the alignment between verbalized confidence and response correctness while maintaining or enhancing overall task performance. These results highlight the potential of semantic perturbation as a practical tool for improving the reliability and interpretability of VLMs. 

**Abstract (ZH)**: 视觉语言模型(VLMs)在多模态任务中表现出色，但经常遭受校准不佳的困扰，导致其口头表达的信心与响应的正确性不一致。这种校准不当会削弱用户的信任，尤其是在模型自信地提供错误或虚假信息时。在本文中，我们提出了一种新的基于语义扰动的信心校准(CSP)框架，以提高VLMs对以物体为中心的查询响应中口头表达的信心的校准。我们首先引入了一个扰动数据集，其中在关键物体区域应用高斯噪声以模拟不同信心水平下的视觉不确定性，建立了视觉模糊性和信心水平之间的明确映射。我们进一步通过结合扰动数据集上的监督微调和后续的偏好优化的两阶段训练过程来增强校准。在流行的基准测试上的广泛实验表明，我们的方法显著提高了口头表达的信心与响应正确性之间的对齐，同时保持或提高了整体任务性能。这些结果突显了语义扰动作为提高VLMs可靠性和可解释性的实际工具的潜力。 

---
# Exploring $\ell_0$ Sparsification for Inference-free Sparse Retrievers 

**Title (ZH)**: 探索基于$\ell_0$稀疏化的技术以实现无推理的稀疏检索 

**Authors**: Xinjie Shen, Zhichao Geng, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14839)  

**Abstract**: With increasing demands for efficiency, information retrieval has developed a branch of sparse retrieval, further advancing towards inference-free retrieval where the documents are encoded during indexing time and there is no model-inference for queries. Existing sparse retrieval models rely on FLOPS regularization for sparsification, while this mechanism was originally designed for Siamese encoders, it is considered to be suboptimal in inference-free scenarios which is asymmetric. Previous attempts to adapt FLOPS for inference-free scenarios have been limited to rule-based methods, leaving the potential of sparsification approaches for inference-free retrieval models largely unexplored. In this paper, we explore $\ell_0$ inspired sparsification manner for inference-free retrievers. Through comprehensive out-of-domain evaluation on the BEIR benchmark, our method achieves state-of-the-art performance among inference-free sparse retrieval models and is comparable to leading Siamese sparse retrieval models. Furthermore, we provide insights into the trade-off between retrieval effectiveness and computational efficiency, demonstrating practical value for real-world applications. 

**Abstract (ZH)**: 基于稀疏性的无推理信息检索方法探究：以BEIR基准全面评估 

---
# Protecting Your Voice: Temporal-aware Robust Watermarking 

**Title (ZH)**: 保护您的声音：基于时间感知的稳健水印技术 

**Authors**: Yue Li, Weizhi Liu, Dongdong Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.14832)  

**Abstract**: The rapid advancement of generative models has led to the synthesis of real-fake ambiguous voices. To erase the ambiguity, embedding watermarks into the frequency-domain features of synthesized voices has become a common routine. However, the robustness achieved by choosing the frequency domain often comes at the expense of fine-grained voice features, leading to a loss of fidelity. Maximizing the comprehensive learning of time-domain features to enhance fidelity while maintaining robustness, we pioneer a \textbf{\underline{t}}emporal-aware \textbf{\underline{r}}ob\textbf{\underline{u}}st wat\textbf{\underline{e}}rmarking (\emph{True}) method for protecting the speech and singing voice. 

**Abstract (ZH)**: 时域aware鲁棒 watermarking (True) 方法：保护语音和唱歌声音的新范式 

---
# ECViT: Efficient Convolutional Vision Transformer with Local-Attention and Multi-scale Stages 

**Title (ZH)**: ECViT: 高效的局部注意和多尺度阶段卷积视觉变换器 

**Authors**: Zhoujie Qian  

**Link**: [PDF](https://arxiv.org/pdf/2504.14825)  

**Abstract**: Vision Transformers (ViTs) have revolutionized computer vision by leveraging self-attention to model long-range dependencies. However, ViTs face challenges such as high computational costs due to the quadratic scaling of self-attention and the requirement of a large amount of training data. To address these limitations, we propose the Efficient Convolutional Vision Transformer (ECViT), a hybrid architecture that effectively combines the strengths of CNNs and Transformers. ECViT introduces inductive biases such as locality and translation invariance, inherent to Convolutional Neural Networks (CNNs) into the Transformer framework by extracting patches from low-level features and enhancing the encoder with convolutional operations. Additionally, it incorporates local-attention and a pyramid structure to enable efficient multi-scale feature extraction and representation. Experimental results demonstrate that ECViT achieves an optimal balance between performance and efficiency, outperforming state-of-the-art models on various image classification tasks while maintaining low computational and storage requirements. ECViT offers an ideal solution for applications that prioritize high efficiency without compromising performance. 

**Abstract (ZH)**: 高效的卷积视觉变换器（ECViT）：结合CNN和Transformer的优势以实现高效性能权衡 

---
# What Lurks Within? Concept Auditing for Shared Diffusion Models at Scale 

**Title (ZH)**: 什么是潜藏其中的？大规模共享扩散模型的概念审查 

**Authors**: Xiaoyong Yuan, Xiaolong Ma, Linke Guo, Lan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14815)  

**Abstract**: Diffusion models (DMs) have revolutionized text-to-image generation, enabling the creation of highly realistic and customized images from text prompts. With the rise of parameter-efficient fine-tuning (PEFT) techniques like LoRA, users can now customize powerful pre-trained models using minimal computational resources. However, the widespread sharing of fine-tuned DMs on open platforms raises growing ethical and legal concerns, as these models may inadvertently or deliberately generate sensitive or unauthorized content, such as copyrighted material, private individuals, or harmful content. Despite the increasing regulatory attention on generative AI, there are currently no practical tools for systematically auditing these models before deployment. In this paper, we address the problem of concept auditing: determining whether a fine-tuned DM has learned to generate a specific target concept. Existing approaches typically rely on prompt-based input crafting and output-based image classification but suffer from critical limitations, including prompt uncertainty, concept drift, and poor scalability. To overcome these challenges, we introduce Prompt-Agnostic Image-Free Auditing (PAIA), a novel, model-centric concept auditing framework. By treating the DM as the object of inspection, PAIA enables direct analysis of internal model behavior, bypassing the need for optimized prompts or generated images. We evaluate PAIA on 320 controlled model and 690 real-world community models sourced from a public DM sharing platform. PAIA achieves over 90% detection accuracy while reducing auditing time by 18-40x compared to existing baselines. To our knowledge, PAIA is the first scalable and practical solution for pre-deployment concept auditing of diffusion models, providing a practical foundation for safer and more transparent diffusion model sharing. 

**Abstract (ZH)**: 扩散模型（DMs）已经革命性地改变了文本到图像的生成，使用户能够从文本提示中生成高度逼真和定制化的图像。随着参数高效微调（PEFT）技术如LoRA的发展，用户现在可以使用最少的计算资源定制强大的预训练模型。然而，广泛共享微调过的DMs在开放平台上引发了日益增长的伦理和法律关切，这些模型可能会无意或故意生成敏感或未经授权的内容，如受版权保护的材料、私人个体或有害内容。尽管监管机构对生成AI给予了越来越多的关注，但目前尚无有效的工具可在部署前系统地审计这些模型。在本文中，我们解决了概念审计的问题：确定微调过的DM是否学会了生成特定的目标概念。现有方法通常依赖于基于提示的输入构建和输出图像分类，但存在提示不确定性、概念漂移及较差的可扩展性等关键局限。为了克服这些挑战，我们引入了提示无关图像无损审计（PAIA）——一种新颖的、以模型为中心的概念审计框架。通过将DM作为检查对象，PAIA可以直接分析模型的内部行为，从而无需优化提示或生成图像即可进行审计。我们在一个公共DM共享平台上收集的真实世界社区模型中评估了PAIA，共涵盖了320个控制模型和690个真实世界的社区模型。PAIA的检测准确率超过90%，同时将审计时间降低了18到40倍，相较于现有基线方法。据我们所知，PAIA是首个可扩展且实用的扩散模型部署前概念审计解决方案，为更安全和透明的扩散模型共享提供了实用的基础。 

---
# On Self-improving Token Embeddings 

**Title (ZH)**: 自改善词嵌入 

**Authors**: Mario M. Kubek, Shiraj Pokharel, Thomas Böhme, Emma L. McDaniel, Herwig Unger, Armin R. Mikler  

**Link**: [PDF](https://arxiv.org/pdf/2504.14808)  

**Abstract**: This article introduces a novel and fast method for refining pre-trained static word or, more generally, token embeddings. By incorporating the embeddings of neighboring tokens in text corpora, it continuously updates the representation of each token, including those without pre-assigned embeddings. This approach effectively addresses the out-of-vocabulary problem, too. Operating independently of large language models and shallow neural networks, it enables versatile applications such as corpus exploration, conceptual search, and word sense disambiguation. The method is designed to enhance token representations within topically homogeneous corpora, where the vocabulary is restricted to a specific domain, resulting in more meaningful embeddings compared to general-purpose pre-trained vectors. As an example, the methodology is applied to explore storm events and their impacts on infrastructure and communities using narratives from a subset of the NOAA Storm Events database. The article also demonstrates how the approach improves the representation of storm-related terms over time, providing valuable insights into the evolving nature of disaster narratives. 

**Abstract (ZH)**: 本文介绍了一种新型且快速的方法，用于细化预训练的静态词嵌入或更一般的令牌嵌入。通过引入文本语料库中相邻令牌的嵌入，该方法持续更新每个令牌的表示，包括那些没有预分配嵌入的令牌。该方法有效地解决了词汇量外问题。该方法独立于大型语言模型和浅层神经网络，使其能够在语料库探索、概念搜索和词义消歧等领域中实现多样化应用。该方法旨在增强在主题同质语料库中令牌的表示，其中词汇量局限于特定领域，从而生成比通用预训练向量更具意义的嵌入。例如，该方法被应用于探索NOAA风暴事件数据库子集中的叙事中风暴事件及其对基础设施和社区的影响。本文还展示了该方法如何随时间提升与风暴相关的术语的表示，提供了有关灾难叙事演变性质的宝贵见解。 

---
# Dynamic Contrastive Skill Learning with State-Transition Based Skill Clustering and Dynamic Length Adjustment 

**Title (ZH)**: 基于状态转换的技能聚类和动态长度调整的动态对比技能学习 

**Authors**: Jinwoo Choi, Seung-Woo Seo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14805)  

**Abstract**: Reinforcement learning (RL) has made significant progress in various domains, but scaling it to long-horizon tasks with complex decision-making remains challenging. Skill learning attempts to address this by abstracting actions into higher-level behaviors. However, current approaches often fail to recognize semantically similar behaviors as the same skill and use fixed skill lengths, limiting flexibility and generalization. To address this, we propose Dynamic Contrastive Skill Learning (DCSL), a novel framework that redefines skill representation and learning. DCSL introduces three key ideas: state-transition based skill representation, skill similarity function learning, and dynamic skill length adjustment. By focusing on state transitions and leveraging contrastive learning, DCSL effectively captures the semantic context of behaviors and adapts skill lengths to match the appropriate temporal extent of behaviors. Our approach enables more flexible and adaptive skill extraction, particularly in complex or noisy datasets, and demonstrates competitive performance compared to existing methods in task completion and efficiency. 

**Abstract (ZH)**: 动态对比技能学习：一种新型的技能表示与学习框架 

---
# Automatic Evaluation Metrics for Document-level Translation: Overview, Challenges and Trends 

**Title (ZH)**: 文档级别翻译的自动评价指标：综述、挑战与趋势 

**Authors**: Jiaxin GUO, Xiaoyu Chen, Zhiqiang Rao, Jinlong Yang, Zongyao Li, Hengchao Shang, Daimeng Wei, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14804)  

**Abstract**: With the rapid development of deep learning technologies, the field of machine translation has witnessed significant progress, especially with the advent of large language models (LLMs) that have greatly propelled the advancement of document-level translation. However, accurately evaluating the quality of document-level translation remains an urgent issue. This paper first introduces the development status of document-level translation and the importance of evaluation, highlighting the crucial role of automatic evaluation metrics in reflecting translation quality and guiding the improvement of translation systems. It then provides a detailed analysis of the current state of automatic evaluation schemes and metrics, including evaluation methods with and without reference texts, as well as traditional metrics, Model-based metrics and LLM-based metrics. Subsequently, the paper explores the challenges faced by current evaluation methods, such as the lack of reference diversity, dependence on sentence-level alignment information, and the bias, inaccuracy, and lack of interpretability of the LLM-as-a-judge method. Finally, the paper looks ahead to the future trends in evaluation methods, including the development of more user-friendly document-level evaluation methods and more robust LLM-as-a-judge methods, and proposes possible research directions, such as reducing the dependency on sentence-level information, introducing multi-level and multi-granular evaluation approaches, and training models specifically for machine translation evaluation. This study aims to provide a comprehensive analysis of automatic evaluation for document-level translation and offer insights into future developments. 

**Abstract (ZH)**: 随着深度学习技术的迅速发展，机器翻译领域取得了显著进步，尤其是大语言模型（LLMs）的出现极大地推动了文档级翻译的发展。然而，准确评估文档级翻译的质量仍然是一个迫切的问题。本文首先介绍了文档级翻译的发展状况及其评估的重要性，突出了自动评估指标在反映翻译质量和指导翻译系统改进中的关键作用。随后，本文详细分析了当前自动评估方案和指标的状态，包括有参考文本和无参考文本的评估方法，以及传统指标、模型基础指标和大语言模型基础指标。接着，本文探讨了现有评估方法面临的挑战，如缺乏参考文本多样性、依赖于句子级对齐信息以及大语言模型作为评估者的偏见、不准确性和可解释性不足。最后，本文展望了评估方法的未来趋势，包括开发更用户友好的文档级评估方法和更 robust 的大语言模型作为评估者的方法，并提出了可能的研究方向，如减少对句子级信息的依赖、引入多级和多粒度的评估方法以及专门训练用于机器翻译评估的模型。本文旨在进行全面的自动评估分析，并为未来的发展提供见解。 

---
# Automated Duplicate Bug Report Detection in Large Open Bug Repositories 

**Title (ZH)**: 大型开放缺陷仓库中自动化重复 Bug 报告检测 

**Authors**: Clare E. Laney, Andrew Barovic, Armin Moin  

**Link**: [PDF](https://arxiv.org/pdf/2504.14797)  

**Abstract**: Many users and contributors of large open-source projects report software defects or enhancement requests (known as bug reports) to the issue-tracking systems. However, they sometimes report issues that have already been reported. First, they may not have time to do sufficient research on existing bug reports. Second, they may not possess the right expertise in that specific area to realize that an existing bug report is essentially elaborating on the same matter, perhaps with a different wording. In this paper, we propose a novel approach based on machine learning methods that can automatically detect duplicate bug reports in an open bug repository based on the textual data in the reports. We present six alternative methods: Topic modeling, Gaussian Naive Bayes, deep learning, time-based organization, clustering, and summarization using a generative pre-trained transformer large language model. Additionally, we introduce a novel threshold-based approach for duplicate identification, in contrast to the conventional top-k selection method that has been widely used in the literature. Our approach demonstrates promising results across all the proposed methods, achieving accuracy rates ranging from the high 70%'s to the low 90%'s. We evaluated our methods on a public dataset of issues belonging to an Eclipse open-source project. 

**Abstract (ZH)**: 基于机器学习方法的开源项目重复bug报告自动检测研究 

---
# How Effective Can Dropout Be in Multiple Instance Learning ? 

**Title (ZH)**: 多实例学习中Dropout的有效性探究 

**Authors**: Wenhui Zhu, Peijie Qiu, Xiwen Chen, Zhangsihao Yang, Aristeidis Sotiras, Abolfazl Razi, Yalin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14783)  

**Abstract**: Multiple Instance Learning (MIL) is a popular weakly-supervised method for various applications, with a particular interest in histological whole slide image (WSI) classification. Due to the gigapixel resolution of WSI, applications of MIL in WSI typically necessitate a two-stage training scheme: first, extract features from the pre-trained backbone and then perform MIL aggregation. However, it is well-known that this suboptimal training scheme suffers from "noisy" feature embeddings from the backbone and inherent weak supervision, hindering MIL from learning rich and generalizable features. However, the most commonly used technique (i.e., dropout) for mitigating this issue has yet to be explored in MIL. In this paper, we empirically explore how effective the dropout can be in MIL. Interestingly, we observe that dropping the top-k most important instances within a bag leads to better performance and generalization even under noise attack. Based on this key observation, we propose a novel MIL-specific dropout method, termed MIL-Dropout, which systematically determines which instances to drop. Experiments on five MIL benchmark datasets and two WSI datasets demonstrate that MIL-Dropout boosts the performance of current MIL methods with a negligible computational cost. The code is available at this https URL. 

**Abstract (ZH)**: 多实例学习（MIL）在各种应用中的弱监督方法，特别是在组织学全视野图像（WSI）分类中的应用受到了广泛关注。由于WSI的 gigapixel 分辨率，MIL 在 WSI 的应用通常需要两阶段训练方案：首先从预训练的骨干网络中提取特征，然后进行 MIL 聚合。然而，众所周知，这种次优训练方案会从骨干网络中产生“噪音”特征嵌入，并且固有的弱监督会阻碍 MIL 学习丰富的可泛化特征。然而，用于减轻这一问题的最常用技术（即丢弃）尚未在 MIL 中进行探索。在本文中，我们实证研究了丢弃在 MIL 中的有效性。有趣的是，我们观察到在包内丢弃最重要的前 k 个实例，即使在噪声攻击下也能提高性能和泛化能力。基于这一关键观察，我们提出了一种新的专用于 MIL 的丢弃方法，称为 MIL-Dropout，该方法系统地确定哪些实例需要被丢弃。在五个 MIL 基准数据集和两个 WSI 数据集上的实验表明，MIL-Dropout 可以以可忽略的计算成本提升现有 MIL 方法的效果。代码可在此处访问：this https URL。 

---
# Exploring Collaborative GenAI Agents in Synchronous Group Settings: Eliciting Team Perceptions and Design Considerations for the Future of Work 

**Title (ZH)**: 探索同步群组环境中协作生成式AI代理：激发团队感知及未来工作设计考虑 

**Authors**: Janet G. Johnson, Macarena Peralta, Mansanjam Kaur, Ruijie Sophia Huang, Sheng Zhao, Ruijia Guan, Shwetha Rajaram, Michael Nebeling  

**Link**: [PDF](https://arxiv.org/pdf/2504.14779)  

**Abstract**: While generative artificial intelligence (GenAI) is finding increased adoption in workplaces, current tools are primarily designed for individual use. Prior work established the potential for these tools to enhance personal creativity and productivity towards shared goals; however, we don't know yet how to best take into account the nuances of group work and team dynamics when deploying GenAI in work settings. In this paper, we investigate the potential of collaborative GenAI agents to augment teamwork in synchronous group settings through an exploratory study that engaged 25 professionals across 6 teams in speculative design workshops and individual follow-up interviews. Our workshops included a mixed reality provotype to simulate embodied collaborative GenAI agents capable of actively participating in group discussions. Our findings suggest that, if designed well, collaborative GenAI agents offer valuable opportunities to enhance team problem-solving by challenging groupthink, bridging communication gaps, and reducing social friction. However, teams' willingness to integrate GenAI agents depended on its perceived fit across a number of individual, team, and organizational factors. We outline the key design tensions around agent representation, social prominence, and engagement and highlight the opportunities spatial and immersive technologies could offer to modulate GenAI influence on team outcomes and strike a balance between augmentation and agency. 

**Abstract (ZH)**: 协作生成人工智能代理在同步团队工作中的潜力探究：基于25名专业人士的探索性研究 

---
# A Combinatorial Theory of Dropout: Subnetworks, Graph Geometry, and Generalization 

**Title (ZH)**: 一种丢弃的组合理论：子网络、图几何与泛化 

**Authors**: Sahil Rajesh Dhayalkar  

**Link**: [PDF](https://arxiv.org/pdf/2504.14762)  

**Abstract**: We propose a combinatorial and graph-theoretic theory of dropout by modeling training as a random walk over a high-dimensional graph of binary subnetworks. Each node represents a masked version of the network, and dropout induces stochastic traversal across this space. We define a subnetwork contribution score that quantifies generalization and show that it varies smoothly over the graph. Using tools from spectral graph theory, PAC-Bayes analysis, and combinatorics, we prove that generalizing subnetworks form large, connected, low-resistance clusters, and that their number grows exponentially with network width. This reveals dropout as a mechanism for sampling from a robust, structured ensemble of well-generalizing subnetworks with built-in redundancy. Extensive experiments validate every theoretical claim across diverse architectures. Together, our results offer a unified foundation for understanding dropout and suggest new directions for mask-guided regularization and subnetwork optimization. 

**Abstract (ZH)**: 我们提出了一种基于组合学和图论的dropout理论，将其训练建模为在高维二元子网络图上的随机游走。每个节点表示网络的一种蒙版版本，dropout 导致在该空间中随机遍历。我们定义了一个子网络贡献得分来量化泛化能力，并证明其在图上平滑变化。利用谱图理论、PAC-Bayes 分析和组合学工具，我们证明了泛化良好的子网络形成了大型、连通且低电阻的聚类，并且其数量随着网络宽度呈指数增长。这揭示了dropout 是从鲁棒且结构化的良好泛化子网络集合中进行采样的机制，该集合具有内置冗余。大量实验在多种架构中验证了每个理论声明。我们的结果为理解dropout 提供了一个统一的基础，并暗示了基于掩码的正则化和子网络优化的新方向。 

---
# SWE-Synth: Synthesizing Verifiable Bug-Fix Data to Enable Large Language Models in Resolving Real-World Bugs 

**Title (ZH)**: SWE-Synth: 合成可验证的bug修复数据以使大型语言模型能够解决实际 bugs 

**Authors**: Minh V.T. Pham, Huy N. Phan, Hoang N. Phan, Cuong Le Chi, Tien N. Nguyen, Nghi D. Q. Bui  

**Link**: [PDF](https://arxiv.org/pdf/2504.14757)  

**Abstract**: Large language models (LLMs) are transforming automated program repair (APR) through agent-based approaches that localize bugs, generate patches, and verify fixes. However, the lack of high-quality, scalable training datasets, especially those with verifiable outputs and intermediate reasoning traces-limits progress, particularly for open-source models. In this work, we present SWE-Synth, a framework for synthesizing realistic, verifiable, and process-aware bug-fix datasets at the repository level. SWE-Synth leverages LLM agents to simulate debugging workflows, producing not only bug-fix pairs but also test cases and structured repair trajectories. Compared to manually curated datasets, our method scales with minimal human effort while preserving contextual richness and correctness. Experiments show that models trained on SWE-Synth outperform those trained on real-world datasets by 2.3% on SWE-Bench Lite. Our results highlight the potential of synthetic, agent-generated data to advance the state of the art in APR and software engineering automation. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过基于代理的方法正在变革自动化程序修复（APR），这些方法用于定位错误、生成补丁并验证修复。然而，高质量、可扩展的训练数据集的缺乏，尤其是那些具有可验证输出和中间推理轨迹的数据集——限制了进展，尤其是在开源模型方面的进展。在本工作中，我们提出了SWE-Synth框架，用于在仓库级别合成现实、可验证且过程意识的漏洞修复数据集。SWE-Synth 利用 LLM 代理来模拟调试工作流程，不仅生成错误修复对，还生成测试案例和结构化的修复轨迹。与手工策展的数据集相比，我们的方法在极少数的人工努力下扩展规模的同时保留了语境丰富性和正确性。实验结果表明，使用 SWE-Synth 训练的模型在 SWE-Bench Lite 上的表现比使用真实世界数据集训练的模型高出 2.3%。我们的结果突显了合成、由代理生成的数据在推动 APR 和软件工程自动化的前沿方面的潜力。 

---
# AI for the Open-World: the Learning Principles 

**Title (ZH)**: 开放世界中的AI：学习原理 

**Authors**: Jianyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14751)  

**Abstract**: During the past decades, numerous successes of AI has been made on "specific capabilities", named closed-world, such as artificial environments or specific real-world tasks. This well-defined narrow capability brings two nice benefits, a clear criterion of success and the opportunity to collect a lot of examples. The criteria not only reveal whether a machine has achieved a goal, but reveal how the machine falls short of the goal. As a result, human designers can fix the problems one after the other until the machine is deemed good enough for the task. Furthermore, the large set of collected examples reduces the difficulty of this problem-fixing process (by the central limit theorem).
Do the success in closed-world translate into broad open-world, where a machine is required to perform any task that a human could possibly undertake with fewer examples and less priori knowledge from human designers? No. Because competence in a specific task provides little insight in handling other tasks, the valuable criteria for specific tasks become helpless when handling broader unseen tasks. Furthermore, due to the shortage of examples in unseen tasks, central limit theorem does not stand on our side. At the end, human designers lose the oscilloscope to "hack" an AI system for the open-world.
Achieving AI for the open-world requires unique learning principles and innovated techniques, which are different from the ones in building AI for the closed-world. This thesis explores necessary learning principles required to construct AI for the open-world, including rich features (analogy a large tool box), disentangled representation (an organized tool box), and inference-time learning (a tool-savvy hand). Driven by the learning principles, this thesis further proposes techniques to use the learning principles, conducts enormous large-scale experiments to verify the learning principles. 

**Abstract (ZH)**: 开放世界中的人工智能学习原则与创新技术 

---
# A Modularized Design Approach for GelSight Family of Vision-based Tactile Sensors 

**Title (ZH)**: 基于视觉触觉传感器GelSight家族的模块化设计方法 

**Authors**: Arpit Agarwal, Mohammad Amin Mirzaee, Xiping Sun, Wenzhen Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2504.14739)  

**Abstract**: GelSight family of vision-based tactile sensors has proven to be effective for multiple robot perception and manipulation tasks. These sensors are based on an internal optical system and an embedded camera to capture the deformation of the soft sensor surface, inferring the high-resolution geometry of the objects in contact. However, customizing the sensors for different robot hands requires a tedious trial-and-error process to re-design the optical system. In this paper, we formulate the GelSight sensor design process as a systematic and objective-driven design problem and perform the design optimization with a physically accurate optical simulation. The method is based on modularizing and parameterizing the sensor's optical components and designing four generalizable objective functions to evaluate the sensor. We implement the method with an interactive and easy-to-use toolbox called OptiSense Studio. With the toolbox, non-sensor experts can quickly optimize their sensor design in both forward and inverse ways following our predefined modules and steps. We demonstrate our system with four different GelSight sensors by quickly optimizing their initial design in simulation and transferring it to the real sensors. 

**Abstract (ZH)**: 基于视觉的GelSight家族触觉传感器在多个机器人感知与 manipulation 任务中证明非常有效。这些传感器基于内部光学系统和嵌入式摄像头以捕获软传感器表面的变形，并推断出接触物体的高分辨率几何结构。然而，针对不同机器人手部定制传感器需要一个繁琐的试错过程来重新设计光学系统。在本文中，我们将GelSight传感器的设计过程转化为一个系统化和目标驱动的设计问题，并利用物理准确的光学模拟进行设计优化。该方法基于模块化和参数化传感器的光学组件，并设计了四个可泛化的目标函数来评估传感器。我们通过一个交互式且易于使用的工具箱OptiSense Studio实施了该方法。利用该工具箱，非传感器专家可以按照我们预定义的模块和步骤，快速从前向和逆向两个方面优化传感器设计。我们通过快速优化四个不同GelSight传感器的初始设计并在仿真实验中进行验证来展示我们的系统，并将优化结果转移到实际传感器上。 

---
# SuperCL: Superpixel Guided Contrastive Learning for Medical Image Segmentation Pre-training 

**Title (ZH)**: SuperCL：基于超像素的对比学习医疗图像分割预训练 

**Authors**: Shuang Zeng, Lei Zhu, Xinliang Zhang, Hangzhou He, Yanye Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14737)  

**Abstract**: Medical image segmentation is a critical yet challenging task, primarily due to the difficulty of obtaining extensive datasets of high-quality, expert-annotated images. Contrastive learning presents a potential but still problematic solution to this issue. Because most existing methods focus on extracting instance-level or pixel-to-pixel representation, which ignores the characteristics between intra-image similar pixel groups. Moreover, when considering contrastive pairs generation, most SOTA methods mainly rely on manually setting thresholds, which requires a large number of gradient experiments and lacks efficiency and generalization. To address these issues, we propose a novel contrastive learning approach named SuperCL for medical image segmentation pre-training. Specifically, our SuperCL exploits the structural prior and pixel correlation of images by introducing two novel contrastive pairs generation strategies: Intra-image Local Contrastive Pairs (ILCP) Generation and Inter-image Global Contrastive Pairs (IGCP) Generation. Considering superpixel cluster aligns well with the concept of contrastive pairs generation, we utilize the superpixel map to generate pseudo masks for both ILCP and IGCP to guide supervised contrastive learning. Moreover, we also propose two modules named Average SuperPixel Feature Map Generation (ASP) and Connected Components Label Generation (CCL) to better exploit the prior structural information for IGCP. Finally, experiments on 8 medical image datasets indicate our SuperCL outperforms existing 12 methods. i.e. Our SuperCL achieves a superior performance with more precise predictions from visualization figures and 3.15%, 5.44%, 7.89% DSC higher than the previous best results on MMWHS, CHAOS, Spleen with 10% annotations. Our code will be released after acceptance. 

**Abstract (ZH)**: 一种用于医学图像分割预训练的新型对比学习方法：SuperCL 

---
# Semi-parametric Memory Consolidation: Towards Brain-like Deep Continual Learning 

**Title (ZH)**: 半参数化记忆巩固： toward 大脑似的深度连续学习 

**Authors**: Geng Liu, Fei Zhu, Rong Feng, Zhiqiang Yi, Shiqi Wang, Gaofeng Meng, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14727)  

**Abstract**: Humans and most animals inherently possess a distinctive capacity to continually acquire novel experiences and accumulate worldly knowledge over time. This ability, termed continual learning, is also critical for deep neural networks (DNNs) to adapt to the dynamically evolving world in open environments. However, DNNs notoriously suffer from catastrophic forgetting of previously learned knowledge when trained on sequential tasks. In this work, inspired by the interactive human memory and learning system, we propose a novel biomimetic continual learning framework that integrates semi-parametric memory and the wake-sleep consolidation mechanism. For the first time, our method enables deep neural networks to retain high performance on novel tasks while maintaining prior knowledge in real-world challenging continual learning scenarios, e.g., class-incremental learning on ImageNet. This study demonstrates that emulating biological intelligence provides a promising path to enable deep neural networks with continual learning capabilities. 

**Abstract (ZH)**: 人类和大多数动物天生具备持续获取新经验和积累 worldly 知识的能力。这种能力称为持续学习，对于在开放环境下适应动态变化世界的深度神经网络（DNNs）也至关重要。然而，当DNNs在顺序任务中训练时，它们 notorious 地会经历对先前学习知识的灾难性遗忘。受交互式人类记忆和学习系统的启发，我们提出了一种新的生物模拟持续学习框架，该框架结合了半参数化记忆和清醒-睡眠巩固机制。我们的方法首次使深度神经网络能够在现实世界的持续学习挑战场景中，如 ImageNet 类增量学习中，保持在新任务上的高性能同时维持先前的知识。本研究显示，模拟生物智能为使深度神经网络具备持续学习能力提供了有前途的道路。 

---
# Exposing the Copycat Problem of Imitation-based Planner: A Novel Closed-Loop Simulator, Causal Benchmark and Joint IL-RL Baseline 

**Title (ZH)**: 基于模仿的学习计划复制问题揭示：一种新型闭环模拟器、因果基准和联合IL-RL基线 

**Authors**: Hui Zhou, Shaoshuai Shi, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.14709)  

**Abstract**: Machine learning (ML)-based planners have recently gained significant attention. They offer advantages over traditional optimization-based planning algorithms. These advantages include fewer manually selected parameters and faster development. Within ML-based planning, imitation learning (IL) is a common algorithm. It primarily learns driving policies directly from supervised trajectory data. While IL has demonstrated strong performance on many open-loop benchmarks, it remains challenging to determine if the learned policy truly understands fundamental driving principles, rather than simply extrapolating from the ego-vehicle's initial state. Several studies have identified this limitation and proposed algorithms to address it. However, these methods often use original datasets for evaluation. In these datasets, future trajectories are heavily dependent on initial conditions. Furthermore, IL often overfits to the most common scenarios. It struggles to generalize to rare or unseen situations.
To address these challenges, this work proposes: 1) a novel closed-loop simulator supporting both imitation and reinforcement learning, 2) a causal benchmark derived from the Waymo Open Dataset to rigorously assess the impact of the copycat problem, and 3) a novel framework integrating imitation learning and reinforcement learning to overcome the limitations of purely imitative approaches. The code for this work will be released soon. 

**Abstract (ZH)**: 基于机器学习的规划器最近受到了广泛关注。它们在参数选择和开发速度方面优于传统的基于优化的规划算法。在基于机器学习的规划中，拟合学习是一种常见的算法，主要从监督轨迹数据中直接学习驾驶策略。尽管在许多开环基准测试中表现出色，但仍然难以确定学习到的策略是否真正理解了基本的驾驶原理，而不仅仅是从ego车辆的初始状态进行外推。已有研究指出了这一局限性，并提出了相应的方法进行解决。然而，这些方法往往使用原始数据集进行评估，这些数据集中的未来轨迹对初始条件有很强的依赖性。此外，拟合学习经常过度拟合最常见的场景，难以泛化到罕见或未见过的情况。为此，本工作提出：1) 一种支持拟合学习和强化学习的新型闭环模拟器，2) 从Waymo Open Dataset派生的一种因果基准，以严格评估复制问题的影响，3) 一种将拟合学习和强化学习集成的新框架，以克服纯拟合方法的局限性。该工作的代码将在不久后发布。 

---
# Time Frequency Analysis of EMG Signal for Gesture Recognition using Fine grained Features 

**Title (ZH)**: 基于细粒度特征的EMG信号时频分析手势识别 

**Authors**: Parshuram N. Aarotale, Ajita Rattani  

**Link**: [PDF](https://arxiv.org/pdf/2504.14708)  

**Abstract**: Electromyography (EMG) based hand gesture recognition converts forearm muscle activity into control commands for prosthetics, rehabilitation, and human computer interaction. This paper proposes a novel approach to EMG-based hand gesture recognition that uses fine-grained classification and presents XMANet, which unifies low-level local and high level semantic cues through cross layer mutual attention among shallow to deep CNN experts. Using stacked spectrograms and scalograms derived from the Short Time Fourier Transform (STFT) and Wavelet Transform (WT), we benchmark XMANet against ResNet50, DenseNet-121, MobileNetV3, and EfficientNetB0. Experimental results on the Grabmyo dataset indicate that, using STFT, the proposed XMANet model outperforms the baseline ResNet50, EfficientNetB0, MobileNetV3, and DenseNet121 models with improvement of approximately 1.72%, 4.38%, 5.10%, and 2.53%, respectively. When employing the WT approach, improvements of around 1.57%, 1.88%, 1.46%, and 2.05% are observed over the same baselines. Similarly, on the FORS EMG dataset, the XMANet(ResNet50) model using STFT shows an improvement of about 5.04% over the baseline ResNet50. In comparison, the XMANet(DenseNet121) and XMANet(MobileNetV3) models yield enhancements of approximately 4.11% and 2.81%, respectively. Moreover, when using WT, the proposed XMANet achieves gains of around 4.26%, 9.36%, 5.72%, and 6.09% over the baseline ResNet50, DenseNet121, MobileNetV3, and EfficientNetB0 models, respectively. These results confirm that XMANet consistently improves performance across various architectures and signal processing techniques, demonstrating the strong potential of fine grained features for accurate and robust EMG classification. 

**Abstract (ZH)**: 基于 electromyography (EMG) 的手部手势识别将前臂肌活动转化为假肢控制命令、康复和人机交互的控制指令。本文提出了一种基于 EMG 的手部手势识别的新方法，并介绍了 XMANet，该方法通过浅层到深层 CNN 专家之间的跨层互注意力机制统一了低级局部和高级语义线索。使用短时傅里叶变换（STFT）和小波变换（WT）得到的堆叠频谱图和小波变化图，我们将 XMANet 与 ResNet50、DenseNet-121、MobileNetV3 和 EfficientNetB0 进行基准测试。实验结果表明，使用 STFT 时，提出的 XMANet 模型分别在基准 ResNet50、EfficientNetB0、MobileNetV3 和 DenseNet121 模型上获得了约 1.72%、4.38%、5.10% 和 2.53% 的性能提升。使用 WT 时，分别获得了约 1.57%、1.88%、1.46% 和 2.05% 的性能提升。同样，在 FORS EMG 数据集上，使用 STFT 的 XMANet(ResNet50) 模型相对于基准 ResNet50 模型提升了约 5.04%。相比之下，XMANet(DenseNet121) 和 XMANet(MobileNetV3) 模型分别获得了约 4.11% 和 2.81% 的提升。此外，使用 WT 时，提出的 XMANet 分别相对于基准 ResNet50、DenseNet121、MobileNetV3 和 EfficientNetB0 模型获得了约 4.26%、9.36%、5.72% 和 6.09% 的性能提升。这些结果证实了 XMANet 在不同架构和信号处理技术下 consistently 提高了性能，展示了精细特征在准确且稳健的 EMG 分类中的强大潜力。 

---
# Can We Ignore Labels In Out of Distribution Detection? 

**Title (ZH)**: 我们可以在分布外检测中忽视标签吗？ 

**Authors**: Hong Yang, Qi Yu, Travis Desel  

**Link**: [PDF](https://arxiv.org/pdf/2504.14704)  

**Abstract**: Out-of-distribution (OOD) detection methods have recently become more prominent, serving as a core element in safety-critical autonomous systems. One major purpose of OOD detection is to reject invalid inputs that could lead to unpredictable errors and compromise safety. Due to the cost of labeled data, recent works have investigated the feasibility of self-supervised learning (SSL) OOD detection, unlabeled OOD detection, and zero shot OOD detection. In this work, we identify a set of conditions for a theoretical guarantee of failure in unlabeled OOD detection algorithms from an information-theoretic perspective. These conditions are present in all OOD tasks dealing with real-world data: I) we provide theoretical proof of unlabeled OOD detection failure when there exists zero mutual information between the learning objective and the in-distribution labels, a.k.a. 'label blindness', II) we define a new OOD task - Adjacent OOD detection - that tests for label blindness and accounts for a previously ignored safety gap in all OOD detection benchmarks, and III) we perform experiments demonstrating that existing unlabeled OOD methods fail under conditions suggested by our label blindness theory and analyze the implications for future research in unlabeled OOD methods. 

**Abstract (ZH)**: 无分布外（OOD）检测方法近年来日益受到关注，成为关键安全自主系统的核心要素。无分布外检测的一个主要目的是拒绝可能导致不可预测错误并威胁安全性的无效输入。由于标签数据的成本较高，近期研究探索了自监督学习（SSL）无分布外检测、未标记的无分布外检测以及零样本无分布外检测的可行性。在本文中，我们从信息论的角度识别出一组理论失败条件，这些条件存在于所有涉及真实世界数据的无分布外任务中：I) 提供理论证明，当学习目标与在分布标签之间不存在互信息时（即“标签盲”），无分布外检测算法会失败，II) 定义一个新的无分布外任务——相邻无分布外检测，以检测标签盲，并弥补所有无分布外检测基准中忽略的安全缺口，III) 进行实验，证明现有的无分布外检测方法在我们的标签盲理论建议的条件下会失败，并分析这对未来无分布外检测方法研究的含义。 

---
# IXGS-Intraoperative 3D Reconstruction from Sparse, Arbitrarily Posed Real X-rays 

**Title (ZH)**: IXGS-手术中从稀疏、任意姿态的真实X射线进行的3D重建 

**Authors**: Sascha Jecklin, Aidana Massalimova, Ruyi Zha, Lilian Calvet, Christoph J. Laux, Mazda Farshad, Philipp Fürnstahl  

**Link**: [PDF](https://arxiv.org/pdf/2504.14699)  

**Abstract**: Spine surgery is a high-risk intervention demanding precise execution, often supported by image-based navigation systems. Recently, supervised learning approaches have gained attention for reconstructing 3D spinal anatomy from sparse fluoroscopic data, significantly reducing reliance on radiation-intensive 3D imaging systems. However, these methods typically require large amounts of annotated training data and may struggle to generalize across varying patient anatomies or imaging conditions. Instance-learning approaches like Gaussian splatting could offer an alternative by avoiding extensive annotation requirements. While Gaussian splatting has shown promise for novel view synthesis, its application to sparse, arbitrarily posed real intraoperative X-rays has remained largely unexplored. This work addresses this limitation by extending the $R^2$-Gaussian splatting framework to reconstruct anatomically consistent 3D volumes under these challenging conditions. We introduce an anatomy-guided radiographic standardization step using style transfer, improving visual consistency across views, and enhancing reconstruction quality. Notably, our framework requires no pretraining, making it inherently adaptable to new patients and anatomies. We evaluated our approach using an ex-vivo dataset. Expert surgical evaluation confirmed the clinical utility of the 3D reconstructions for navigation, especially when using 20 to 30 views, and highlighted the standardization's benefit for anatomical clarity. Benchmarking via quantitative 2D metrics (PSNR/SSIM) confirmed performance trade-offs compared to idealized settings, but also validated the improvement gained from standardization over raw inputs. This work demonstrates the feasibility of instance-based volumetric reconstruction from arbitrary sparse-view X-rays, advancing intraoperative 3D imaging for surgical navigation. 

**Abstract (ZH)**: 基于实例的学习方法在稀疏视图X射线三维重建中的应用：改进脊柱手术导航成像 

---
# Learning Critically: Selective Self Distillation in Federated Learning on Non-IID Data 

**Title (ZH)**: 批判性学习：联邦学习中非 IID 数据的选择性自我精炼 

**Authors**: Yuting He, Yiqiang Chen, XiaoDong Yang, Hanchao Yu, Yi-Hua Huang, Yang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14694)  

**Abstract**: Federated learning (FL) enables multiple clients to collaboratively train a global model while keeping local data decentralized. Data heterogeneity (non-IID) across clients has imposed significant challenges to FL, which makes local models re-optimize towards their own local optima and forget the global knowledge, resulting in performance degradation and convergence slowdown. Many existing works have attempted to address the non-IID issue by adding an extra global-model-based regularizing item to the local training but without an adaption scheme, which is not efficient enough to achieve high performance with deep learning models. In this paper, we propose a Selective Self-Distillation method for Federated learning (FedSSD), which imposes adaptive constraints on the local updates by self-distilling the global model's knowledge and selectively weighting it by evaluating the credibility at both the class and sample level. The convergence guarantee of FedSSD is theoretically analyzed and extensive experiments are conducted on three public benchmark datasets, which demonstrates that FedSSD achieves better generalization and robustness in fewer communication rounds, compared with other state-of-the-art FL methods. 

**Abstract (ZH)**: federated学习（FL）使多个客户端能够在保持本地数据分散的情况下协作训练全局模型。客户端之间数据异质性（非IID）对FL造成了重大挑战，这使得本地模型重新优化以适应自己的局部最优解并遗忘全局知识，导致性能下降和收敛速度变慢。许多现有工作试图通过在本地训练中添加一个基于全局模型的正则化项来解决非IID问题，但缺乏适应性方案，这在使用深度学习模型时效率不够高以达到高性能。本文提出了一种适用于 federated学习的可选择自/distillation方法（FedSSD），通过自我蒸馏全局模型的知识并在评估类别和样本层面的可信度后选择性加权来对局部更新施加适应性约束。从理论上分析了FedSSD的收敛保证，并在三个开源基准数据集上进行了广泛的实验，实验结果表明，与当前最先进的FL方法相比，FedSSD在较少的通信轮次中实现了更好的泛化能力和鲁棒性。 

---
# Video-MMLU: A Massive Multi-Discipline Lecture Understanding Benchmark 

**Title (ZH)**: Video-MMLU: 一个大规模多学科讲座理解基准 

**Authors**: Enxin Song, Wenhao Chai, Weili Xu, Jianwen Xie, Yuxuan Liu, Gaoang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14693)  

**Abstract**: Recent advancements in language multimodal models (LMMs) for video have demonstrated their potential for understanding video content, yet the task of comprehending multi-discipline lectures remains largely unexplored. We introduce Video-MMLU, a massive benchmark designed to evaluate the capabilities of LMMs in understanding Multi-Discipline Lectures. We evaluate over 90 open-source and proprietary models, ranging from 0.5B to 40B parameters. Our results highlight the limitations of current models in addressing the cognitive challenges presented by these lectures, especially in tasks requiring both perception and reasoning. Additionally, we explore how the number of visual tokens and the large language models influence performance, offering insights into the interplay between multimodal perception and reasoning in lecture comprehension. 

**Abstract (ZH)**: 近期语言多模态模型（LMMs）在视频理解领域的进展展示了其潜在的应用价值，但多学科讲座的理解任务仍 largely unexplored。我们介绍了 Video-MMLU，一个大规模基准，旨在评估 LMMs 在理解多学科讲座方面的能力。我们评估了超过 90 个开源和专有模型，参数量从 0.5B 到 40B 不等。我们的结果突显了当前模型在解决这些讲座带来的认知挑战方面的局限性，尤其是在需要感知和推理相结合的任务中。此外，我们还探讨了视觉标记的数量和大规模语言模型对性能的影响，提供了多模态感知与讲座理解中推理之间的相互作用的见解。 

---
# FarsEval-PKBETS: A new diverse benchmark for evaluating Persian large language models 

**Title (ZH)**: FarsEval-PKBETS：一个新的多样化基准，用于评估波斯大型语言模型 

**Authors**: Mehrnoush Shamsfard, Zahra Saaberi, Mostafa Karimi manesh, Seyed Mohammad Hossein Hashemi, Zahra Vatankhah, Motahareh Ramezani, Niki Pourazin, Tara Zare, Maryam Azimi, Sarina Chitsaz, Sama Khoraminejad, Morteza Mahdavi Mortazavi, Mohammad Mahdi Chizari, Sahar Maleki, Seyed Soroush Majd, Mostafa Masumi, Sayed Ali Musavi Khoeini, Amir Mohseni, Sogol Alipour  

**Link**: [PDF](https://arxiv.org/pdf/2504.14690)  

**Abstract**: Research on evaluating and analyzing large language models (LLMs) has been extensive for resource-rich languages such as English, yet their performance in languages such as Persian has received considerably less attention. This paper introduces FarsEval-PKBETS benchmark, a subset of FarsEval project for evaluating large language models in Persian. This benchmark consists of 4000 questions and answers in various formats, including multiple choice, short answer and descriptive responses. It covers a wide range of domains and tasks,including medicine, law, religion, Persian language, encyclopedic knowledge, human preferences, social knowledge, ethics and bias, text generation, and respecting others' rights. This bechmark incorporates linguistics, cultural, and local considerations relevant to the Persian language and Iran. To ensure the questions are challenging for current LLMs, three models -- Llama3-70B, PersianMind, and Dorna -- were evaluated using this benchmark. Their average accuracy was below 50%, meaning they provided fully correct answers to fewer than half of the questions. These results indicate that current language models are still far from being able to solve this benchmark 

**Abstract (ZH)**: 对大语言模型（LLMs）在波斯语中的评估与分析研究已经很广泛，尤其是在资源丰富的语言例如英语方面，但波斯语等语言方面的表现则获得了较少的关注。本文介绍了FarsEval-PKBETS基准，它是FarsEval项目中用于评估波斯语大语言模型的子集。该基准包含4000个采用多种格式的问题和答案，包括选择题、简答题和描述性回答。涵盖的领域和任务包括医学、法律、宗教、波斯语、百科知识、人类偏好、社会知识、伦理和偏见、文本生成以及尊重他人权利等。该基准结合了与波斯语和伊朗相关的语言学、文化及当地考虑因素。为了确保问题对当前的LLMs具有挑战性，使用此基准评估了三种模型——Llama3-70B、PersianMind和Dorna。它们的平均准确率低于50%，意味着它们能够给出完整正确答案的问题不到一半。这些结果表明，当前的语言模型距离能够解决此基准任务还有很长的路要走。 

---
# Uncovering Issues in the Radio Access Network by Looking at the Neighbors 

**Title (ZH)**: 通过观察邻区发现无线接入网络中的问题 

**Authors**: José Suárez-Varela, Andra Lutu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14686)  

**Abstract**: Mobile network operators (MNOs) manage Radio Access Networks (RANs) with massive amounts of cells over multiple radio generations (2G-5G). To handle such complexity, operations teams rely on monitoring systems, including anomaly detection tools that identify unexpected behaviors. In this paper, we present c-ANEMON, a Contextual ANomaly dEtection MONitor for the RAN based on Graph Neural Networks (GNNs). Our solution captures spatio-temporal variations by analyzing the behavior of individual cells in relation to their local neighborhoods, enabling the detection of anomalies that are independent of external mobility factors. This, in turn, allows focusing on anomalies associated with network issues (e.g., misconfigurations, equipment failures). We evaluate c-ANEMON using real-world data from a large European metropolitan area (7,890 cells; 3 months). First, we show that the GNN model within our solution generalizes effectively to cells from previously unseen areas, suggesting the possibility of using a single model across extensive deployment regions. Then, we analyze the anomalies detected by c-ANEMON through manual inspection and define several categories of long-lasting anomalies (6+ hours). Notably, 45.95% of these anomalies fall into a category that is more likely to require intervention by operations teams. 

**Abstract (ZH)**: 基于图形神经网络的RAN上下文异常检测监控c-ANEMON 

---
# An LLM-enabled Multi-Agent Autonomous Mechatronics Design Framework 

**Title (ZH)**: 一种基于LLM的多agent自主机电设计框架 

**Authors**: Zeyu Wang, Frank P.-W. Lo, Qian Chen, Yongqi Zhang, Chen Lin, Xu Chen, Zhenhua Yu, Alexander J. Thompson, Eric M. Yeatman, Benny P. L. Lo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14681)  

**Abstract**: Existing LLM-enabled multi-agent frameworks are predominantly limited to digital or simulated environments and confined to narrowly focused knowledge domain, constraining their applicability to complex engineering tasks that require the design of physical embodiment, cross-disciplinary integration, and constraint-aware reasoning. This work proposes a multi-agent autonomous mechatronics design framework, integrating expertise across mechanical design, optimization, electronics, and software engineering to autonomously generate functional prototypes with minimal direct human design input. Operating primarily through a language-driven workflow, the framework incorporates structured human feedback to ensure robust performance under real-world constraints. To validate its capabilities, the framework is applied to a real-world challenge involving autonomous water-quality monitoring and sampling, where traditional methods are labor-intensive and ecologically disruptive. Leveraging the proposed system, a fully functional autonomous vessel was developed with optimized propulsion, cost-effective electronics, and advanced control. The design process was carried out by specialized agents, including a high-level planning agent responsible for problem abstraction and dedicated agents for structural, electronics, control, and software development. This approach demonstrates the potential of LLM-based multi-agent systems to automate real-world engineering workflows and reduce reliance on extensive domain expertise. 

**Abstract (ZH)**: 现有的基于大语言模型的多 agents 框架主要局限于数字或模拟环境，并且局限于狭窄的知识领域，限制了其在需要设计物理实体、跨学科集成和约束aware推理的复杂工程任务中的应用。本工作提出了一种多 agents 自主机电设计框架，将机械设计、优化、电子和软件工程方面的专业知识整合起来，以最少的人工设计输入自动生成功能原型。该框架主要通过语言驱动的工作流运行，并整合结构化的人类反馈以确保在实际约束下的稳健性能。为了验证其能力，该框架应用于一项涉及自主水质监测和采样的真实世界挑战，其中传统方法劳动密集且生态破坏性。利用提出系统，开发了一种具有优化推进、低成本电子设备和高级控制的全功能自主船舶。设计过程由专门的代理执行，包括负责问题抽象的高级规划代理和专门的结构设计、电子设备、控制和软件开发代理。这种方法展示了基于大语言模型的多 agents 系统在自动化真实世界工程工作流和减少对广泛领域专业知识依赖方面的潜力。 

---
# Evaluating Temporal Plasticity in Foundation Time Series Models for Incremental Fine-tuning 

**Title (ZH)**: 基础时间序列模型的增量Fine-tuning中的时间可塑性评估 

**Authors**: Jia Liu, Cheng Jinguo, Xia Fang, Zhenyuan Ma, Yuankai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14677)  

**Abstract**: Time series foundation models excel at diverse time series forecasting tasks, but their capacity for continuous improvement through incremental learning remains unexplored. We present the first comprehensive study investigating these models' temporal plasticity - their ability to progressively enhance performance through continual learning while maintaining existing capabilities. Through experiments on real-world datasets exhibiting distribution shifts, we evaluate both conventional deep learning models and foundation models using a novel continual learning framework. Our findings reveal that while traditional models struggle with performance deterioration during incremental fine-tuning, foundation models like Time-MoE and Chronos demonstrate sustained improvement in predictive accuracy. This suggests that optimizing foundation model fine-tuning strategies may be more valuable than developing domain-specific small models. Our research introduces new evaluation methodologies and insights for developing foundation time series models with robust continuous learning capabilities. 

**Abstract (ZH)**: 时间序列基础模型在多样化的时序预测任务中表现出色，但其通过增量学习进行持续改进的能力尚未被探索。我们首次对这些模型的时间灵活性进行了全面研究——它们在不断学习以逐步提升性能的同时，能够保持现有能力。通过在表现出分布偏移的现实数据集上进行实验，我们使用新颖的增量学习框架评估了传统深度学习模型和基础模型的性能。我们的发现表明，与传统模型在增量微调过程中性能下降的情况不同，如Time-MoE和Chronos等基础模型展示了预测准确性的持续提升。这表明，优化基础模型的增量微调策略可能比开发针对特定领域的小型模型更有价值。我们的研究为开发具备稳健持续学习能力的基础时间序列模型引入了新的评估方法和见解。 

---
# A Case Study Exploring the Current Landscape of Synthetic Medical Record Generation with Commercial LLMs 

**Title (ZH)**: 一个案例研究：探索商用大语言模型生成合成医疗记录的当前格局 

**Authors**: Yihan Lin, Zhirong Bella Yu, Simon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.14657)  

**Abstract**: Synthetic Electronic Health Records (EHRs) offer a valuable opportunity to create privacy preserving and harmonized structured data, supporting numerous applications in healthcare. Key benefits of synthetic data include precise control over the data schema, improved fairness and representation of patient populations, and the ability to share datasets without concerns about compromising real individuals privacy. Consequently, the AI community has increasingly turned to Large Language Models (LLMs) to generate synthetic data across various domains. However, a significant challenge in healthcare is ensuring that synthetic health records reliably generalize across different hospitals, a long standing issue in the field. In this work, we evaluate the current state of commercial LLMs for generating synthetic data and investigate multiple aspects of the generation process to identify areas where these models excel and where they fall short. Our main finding from this work is that while LLMs can reliably generate synthetic health records for smaller subsets of features, they struggle to preserve realistic distributions and correlations as the dimensionality of the data increases, ultimately limiting their ability to generalize across diverse hospital settings. 

**Abstract (ZH)**: 合成电子健康记录（EHRs）提供了创建隐私保护和和谐结构化数据的宝贵机会，支持医疗保健领域的众多应用。合成数据的关键优势包括对数据模式的精确控制、改善患者群体的公平性和代表性，以及无需担心泄露实际个体隐私即可共享数据集的能力。因此，AI社区越来越多地转向大型语言模型（LLMs）来生成各种领域的合成数据。然而，在医疗保健领域，确保合成健康记录能够可靠地跨不同医院泛化是一个长期存在的问题。在这项工作中，我们评估了当前商用LLMs生成合成数据的状态，并调查生成过程的多个方面，以确定这些模型的优势和不足之处。我们的主要发现是，虽然LLMs能够可靠地为较小的特征子集生成合成健康记录，但它们在数据维度增加时难以保持现实分布和相关性，最终限制了它们跨多样化的医院环境泛化的能力。 

---
# Surrogate Fitness Metrics for Interpretable Reinforcement Learning 

**Title (ZH)**: 可解释强化学习的代理 fitness 度量标准 

**Authors**: Philipp Altmann, Céline Davignon, Maximilian Zorn, Fabian Ritz, Claudia Linnhoff-Popien, Thomas Gabor  

**Link**: [PDF](https://arxiv.org/pdf/2504.14645)  

**Abstract**: We employ an evolutionary optimization framework that perturbs initial states to generate informative and diverse policy demonstrations. A joint surrogate fitness function guides the optimization by combining local diversity, behavioral certainty, and global population diversity. To assess demonstration quality, we apply a set of evaluation metrics, including the reward-based optimality gap, fidelity interquartile means (IQMs), fitness composition analysis, and trajectory visualizations. Hyperparameter sensitivity is also examined to better understand the dynamics of trajectory optimization. Our findings demonstrate that optimizing trajectory selection via surrogate fitness metrics significantly improves interpretability of RL policies in both discrete and continuous environments. In gridworld domains, evaluations reveal significantly enhanced demonstration fidelities compared to random and ablated baselines. In continuous control, the proposed framework offers valuable insights, particularly for early-stage policies, while fidelity-based optimization proves more effective for mature policies. By refining and systematically analyzing surrogate fitness functions, this study advances the interpretability of RL models. The proposed improvements provide deeper insights into RL decision-making, benefiting applications in safety-critical and explainability-focused domains. 

**Abstract (ZH)**: 我们采用一种进化优化框架，通过扰动初始状态生成具有信息性和多样性的策略演示。联合代理适应性函数通过结合局部多样性、行为确定性和全局种群多样性来指导优化。为了评估演示质量，我们应用了包括基于奖励的最优性差距、保真度四分位均值（IQMs）、适应性组成分析和轨迹可视化在内的一系列评估指标。我们还研究了超参数敏感性，以更好地理解轨迹优化的动力学。研究结果表明，通过代理适应性度量优化轨迹选择显著提高了离散和连续环境中的RL策略的可解释性。在格网世界领域，评估结果显示与随机和删减基准相比，演示保真度有显著提升。在连续控制中，所提出的框架对于早期策略提供了有价值的见解，而基于保真度的优化对于成熟策略更为有效。通过细化和系统分析代理适应性函数，本研究推进了RL模型的可解释性。提出的改进为安全关键和可解释性导向领域的RL决策提供了更深入的洞见。 

---
# Risk Assessment Framework for Code LLMs via Leveraging Internal States 

**Title (ZH)**: 基于利用内部状态的风险评估框架：面向代码LLMs 

**Authors**: Yuheng Huang, Lei Ma, Keizaburo Nishikino, Takumi Akazaki  

**Link**: [PDF](https://arxiv.org/pdf/2504.14640)  

**Abstract**: The pre-training paradigm plays a key role in the success of Large Language Models (LLMs), which have been recognized as one of the most significant advancements of AI recently. Building on these breakthroughs, code LLMs with advanced coding capabilities bring huge impacts on software engineering, showing the tendency to become an essential part of developers' daily routines. However, the current code LLMs still face serious challenges related to trustworthiness, as they can generate incorrect, insecure, or unreliable code. Recent exploratory studies find that it can be promising to detect such risky outputs by analyzing LLMs' internal states, akin to how the human brain unconsciously recognizes its own mistakes. Yet, most of these approaches are limited to narrow sub-domains of LLM operations and fall short of achieving industry-level scalability and practicability. To address these challenges, in this paper, we propose PtTrust, a two-stage risk assessment framework for code LLM based on internal state pre-training, designed to integrate seamlessly with the existing infrastructure of software companies. The core idea is that the risk assessment framework could also undergo a pre-training process similar to LLMs. Specifically, PtTrust first performs unsupervised pre-training on large-scale unlabeled source code to learn general representations of LLM states. Then, it uses a small, labeled dataset to train a risk predictor. We demonstrate the effectiveness of PtTrust through fine-grained, code line-level risk assessment and demonstrate that it generalizes across tasks and different programming languages. Further experiments also reveal that PtTrust provides highly intuitive and interpretable features, fostering greater user trust. We believe PtTrust makes a promising step toward scalable and trustworthy assurance for code LLMs. 

**Abstract (ZH)**: 基于内部状态预训练的代码LLM风险评估框架PtTrust 

---
# AlphaZero-Edu: Making AlphaZero Accessible to Everyone 

**Title (ZH)**: AlphaZero-Edu: 让AlphaZero触达每一个人 

**Authors**: Binjie Guo, Hanyu Zheng, Guowei Su, Ru Zhang, Haohan Jiang, Xurong Lin, Hongyan Wei, Aisheng Mo, Jie Li, Zhiyuan Qian, Zhuhao Zhang, Xiaoyuan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.14636)  

**Abstract**: Recent years have witnessed significant progress in reinforcement learning, especially with Zero-like paradigms, which have greatly boosted the generalization and reasoning abilities of large-scale language models. Nevertheless, existing frameworks are often plagued by high implementation complexity and poor reproducibility. To tackle these challenges, we present AlphaZero-Edu, a lightweight, education-focused implementation built upon the mathematical framework of AlphaZero. It boasts a modular architecture that disentangles key components, enabling transparent visualization of the algorithmic processes. Additionally, it is optimized for resource-efficient training on a single NVIDIA RTX 3090 GPU and features highly parallelized self-play data generation, achieving a 3.2-fold speedup with 8 processes. In Gomoku matches, the framework has demonstrated exceptional performance, achieving a consistently high win rate against human opponents. AlphaZero-Edu has been open-sourced at this https URL, providing an accessible and practical benchmark for both academic research and industrial applications. 

**Abstract (ZH)**: Recent Years Have Witnessed Significant Progress in Reinforcement Learning, Especially with Zero-like Paradigms, Which Have Greatly Boosted the Generalization and Reasoning Abilities of Large-scale Language Models. Nevertheless, Existing Frameworks Are Often Plagued by High Implementation Complexity and Poor Reproducibility. To Tackle These Challenges, We Present AlphaZero-Edu, a Lightweight, Education-Focused Implementation Built Upon the Mathematical Framework of AlphaZero. It Boasts a Modular Architecture That Disentangles Key Components, Enabling Transparent Visualization of the Algorithmic Processes. Additionally, It Is Optimized for Resource-Efficient Training on a Single NVIDIA RTX 3090 GPU and Features Highly Parallelized Self-Play Data Generation, Achieving a 3.2-Fold Speedup With 8 Processes. In Gomoku Matches, the Framework Has Demonstrated Exceptional Performance, Achieving a Consistently High Win Rate Against Human Opponents. AlphaZero-Edu Has Been Open-Sourced at This https URL, Providing an Accessible and Practical Benchmark for Both Academic Research and Industrial Applications. 

---
# Towards Optimal Circuit Generation: Multi-Agent Collaboration Meets Collective Intelligence 

**Title (ZH)**: 向着最优电路生成：多智能体协作与集体智能的融合 

**Authors**: Haiyan Qin, Jiahao Feng, Xiaotong Feng, Wei W. Xing, Wang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14625)  

**Abstract**: Large language models (LLMs) have transformed code generation, yet their application in hardware design produces gate counts 38\%--1075\% higher than human designs. We present CircuitMind, a multi-agent framework that achieves human-competitive efficiency through three key innovations: syntax locking (constraining generation to basic logic gates), retrieval-augmented generation (enabling knowledge-driven design), and dual-reward optimization (balancing correctness with efficiency). To evaluate our approach, we introduce TC-Bench, the first gate-level benchmark harnessing collective intelligence from the TuringComplete ecosystem -- a competitive circuit design platform with hundreds of thousands of players. Experiments show CircuitMind enables 55.6\% of model implementations to match or exceed top-tier human experts in composite efficiency metrics. Most remarkably, our framework elevates the 14B Phi-4 model to outperform both GPT-4o mini and Gemini 2.0 Flash, achieving efficiency comparable to the top 25\% of human experts without requiring specialized training. These innovations establish a new paradigm for hardware optimization where collaborative AI systems leverage collective human expertise to achieve optimal circuit designs. Our model, data, and code are open-source at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）已在代码生成领域产生革命性影响，但在硬件设计中的应用却导致门电路数量比人工设计高出38%至1075%。我们提出了CircuitMind多智能体框架，通过三项关键创新实现与人类竞争的效率：语法锁定（约束生成到基本逻辑门）、检索增强生成（实现知识驱动设计）和双重奖励优化（平衡正确性和效率）。为了评估我们的方法，我们引入了TC-Bench基准测试，这是首个利用图灵完备生态系统集体智慧的门级基准测试——一个拥有数十万参赛者的竞争性电路设计平台。实验结果显示，CircuitMind使55.6%的模型实现能够匹配或超越顶级人工专家的综合效率指标。尤为令人瞩目的是，我们的框架将14B Phi-4模型提升到在效率上优于GPT-4o mini和Gemini 2.0 Flash，无需专门训练即可达到顶级人工专家前25%的效率水平。这些创新确立了一个新的硬件优化范式，即协作式AI系统利用集体人类专业知识来实现最佳电路设计。我们的模型、数据和代码已开源。 

---
# VM-BHINet:Vision Mamba Bimanual Hand Interaction Network for 3D Interacting Hand Mesh Recovery From a Single RGB Image 

**Title (ZH)**: VM-BHINet: Vision Mamba Bimanual Hand Interaction Network for 3D Interacting Hand Mesh Recovery from a Single RGB Image 

**Authors**: Han Bi, Ge Yu, Yu He, Wenzhuo Liu, Zijie Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.14618)  

**Abstract**: Understanding bimanual hand interactions is essential for realistic 3D pose and shape reconstruction. However, existing methods struggle with occlusions, ambiguous appearances, and computational inefficiencies. To address these challenges, we propose Vision Mamba Bimanual Hand Interaction Network (VM-BHINet), introducing state space models (SSMs) into hand reconstruction to enhance interaction modeling while improving computational efficiency. The core component, Vision Mamba Interaction Feature Extraction Block (VM-IFEBlock), combines SSMs with local and global feature operations, enabling deep understanding of hand interactions. Experiments on the InterHand2.6M dataset show that VM-BHINet reduces Mean per-joint position error (MPJPE) and Mean per-vertex position error (MPVPE) by 2-3%, significantly surpassing state-of-the-art methods. 

**Abstract (ZH)**: 理解双手中手交互对于实现逼真的3D姿态和形状重建是必不可少的。然而，现有方法在处理遮挡、模糊外观和计算效率低下方面存在困难。为了解决这些挑战，我们提出了Vision Mamba双手中手交互网络（VM-BHINet），通过引入状态空间模型（SSMs）来增强交互建模并提高计算效率。核心组件，Vision Mamba交互特征提取块（VM-IFEBlock），将SSMs与局部和全局特征操作相结合，实现了对手交互的深层次理解。实验结果显示，VM-BHINet在InterHand2.6M数据集上的Mean per-joint position error（MPJPE）和Mean per-vertex position error（MPVPE）分别减少了2-3%，显著优于现有最佳方法。 

---
# K2MUSE: A human lower limb multimodal dataset under diverse conditions for facilitating rehabilitation robotics 

**Title (ZH)**: K2MUSE：在多样条件下的人类下肢多模态数据集，以促进康复 robotics 的发展 

**Authors**: Jiwei Li, Bi Zhang, Xiaowei Tan, Wanxin Chen, Zhaoyuan Liu, Juanjuan Zhang, Weiguang Huo, Jian Huang, Lianqing Liu, Xingang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.14602)  

**Abstract**: The natural interaction and control performance of lower limb rehabilitation robots are closely linked to biomechanical information from various human locomotion activities. Multidimensional human motion data significantly deepen the understanding of the complex mechanisms governing neuromuscular alterations, thereby facilitating the development and application of rehabilitation robots in multifaceted real-world environments. However, currently available lower limb datasets are inadequate for supplying the essential multimodal data and large-scale gait samples necessary for effective data-driven approaches, and they neglect the significant effects of acquisition interference in real this http URL fill this gap, we present the K2MUSE dataset, which includes a comprehensive collection of multimodal data, comprising kinematic, kinetic, amplitude-mode ultrasound (AUS), and surface electromyography (sEMG) measurements. The proposed dataset includes lower limb multimodal data from 30 able-bodied participants walking under different inclines (0$^\circ$, $\pm$5$^\circ$, and $\pm$10$^\circ$), various speeds (0.5 m/s, 1.0 m/s, and 1.5 m/s), and different nonideal acquisition conditions (muscle fatigue, electrode shifts, and inter-day differences). The kinematic and ground reaction force data were collected via a Vicon motion capture system and an instrumented treadmill with embedded force plates, whereas the sEMG and AUS data were synchronously recorded for thirteen muscles on the bilateral lower limbs. This dataset offers a new resource for designing control frameworks for rehabilitation robots and conducting biomechanical analyses of lower limb locomotion. The dataset is available at this https URL. 

**Abstract (ZH)**: 下肢康复机器人的人机自然交互与控制性能与各种人类运动活动的生物力学信息密切相关。多维度的人体运动数据大大加深了对调控神经肌肉转换复杂机制的理解，从而促进了康复机器人在多场景实际环境中的开发与应用。然而，目前可用的下肢数据集不足以提供有效数据驱动方法所需的各种模态数据和大规模步态样本，并且忽略了实际获取干扰的显著影响。为填补这一空白，我们提出了K2MUSE数据集，该数据集包含全面的多模态数据，包括运动学、动力学、幅度模式超声波（AUS）和表面肌电图（sEMG）测量。所提出的数据集包括30名健康受试者在不同坡度（0°，±5°，±10°）、不同速度（0.5 m/s，1.0 m/s，1.5 m/s）和不同非理想采集条件下（肌肉疲劳、电极位移和日间差异）的下肢多模态数据。运动学和地面反作用力数据通过Vicon运动捕捉系统和内置力板的仪器跑步机收集，而sEMG和AUS数据同步记录了双侧下肢的十三块肌肉。该数据集为设计康复机器人控制框架和进行下肢单位运动的生物力学分析提供了新的资源。数据集可从此链接获得。 

---
# HealthGenie: Empowering Users with Healthy Dietary Guidance through Knowledge Graph and Large Language Models 

**Title (ZH)**: 健康 genie: 通过知识图谱和大语言模型赋能用户的健康饮食指导 

**Authors**: Fan Gao, Xinjie Zhao, Ding Xia, Zhongyi Zhou, Rui Yang, Jinghui Lu, Hang Jiang, Chanjun Park, Irene Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.14594)  

**Abstract**: Seeking dietary guidance often requires navigating complex professional knowledge while accommodating individual health conditions. Knowledge Graphs (KGs) offer structured and interpretable nutritional information, whereas Large Language Models (LLMs) naturally facilitate conversational recommendation delivery. In this paper, we present HealthGenie, an interactive system that combines the strengths of LLMs and KGs to provide personalized dietary recommendations along with hierarchical information visualization for a quick and intuitive overview. Upon receiving a user query, HealthGenie performs query refinement and retrieves relevant information from a pre-built KG. The system then visualizes and highlights pertinent information, organized by defined categories, while offering detailed, explainable recommendation rationales. Users can further tailor these recommendations by adjusting preferences interactively. Our evaluation, comprising a within-subject comparative experiment and an open-ended discussion, demonstrates that HealthGenie effectively supports users in obtaining personalized dietary guidance based on their health conditions while reducing interaction effort and cognitive load. These findings highlight the potential of LLM-KG integration in supporting decision-making through explainable and visualized information. We examine the system's usefulness and effectiveness with an N=12 within-subject study and provide design considerations for future systems that integrate conversational LLM and KG. 

**Abstract (ZH)**: 寻求饮食指导往往需要在复杂的专业知识和个体健康状况之间进行权衡。知识图谱（KGs）提供了结构化和可解释的营养信息，而大型语言模型（LLMs）则自然地促进了对话推荐的交付。本文介绍了一种名为HealthGenie的交互系统，该系统结合了LLMs和KGs的优势，提供个性化饮食建议，并通过层次信息可视化提供快速直观的概览。收到用户查询后，HealthGenie进行查询精炼，并从预构建的知识图谱中检索相关信息。系统随后通过定义的类别组织和突出显示相关信息，同时提供详细的、可解释的推荐理由。用户可以通过交互方式进一步调整这些建议。我们的评估包括一个单被试对照实验和一个开放式讨论，表明HealthGenie能够有效地支持用户根据个人健康状况获得个性化饮食指导，同时减少交互努力和认知负担。这些发现突显了LLM-KG集成在通过可解释和可视化信息支持决策方面的潜力。我们通过N=12的单被试研究评估了该系统的实用性和有效性，并提出了未来结合对话LLM和知识图谱的系统的界面设计考虑。 

---
# Phoenix: A Motion-based Self-Reflection Framework for Fine-grained Robotic Action Correction 

**Title (ZH)**: Phoenix：基于运动的细粒度机器人动作修正自我反思框架 

**Authors**: Wenke Xia, Ruoxuan Feng, Dong Wang, Di Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14588)  

**Abstract**: Building a generalizable self-correction system is crucial for robots to recover from failures. Despite advancements in Multimodal Large Language Models (MLLMs) that empower robots with semantic reflection ability for failure, translating semantic reflection into how to correct fine-grained robotic actions remains a significant challenge. To address this gap, we build the Phoenix framework, which leverages motion instruction as a bridge to connect high-level semantic reflection with low-level robotic action correction. In this motion-based self-reflection framework, we start with a dual-process motion adjustment mechanism with MLLMs to translate the semantic reflection into coarse-grained motion instruction adjustment. To leverage this motion instruction for guiding how to correct fine-grained robotic actions, a multi-task motion-conditioned diffusion policy is proposed to integrate visual observations for high-frequency robotic action correction. By combining these two models, we could shift the demand for generalization capability from the low-level manipulation policy to the MLLMs-driven motion adjustment model and facilitate precise, fine-grained robotic action correction. Utilizing this framework, we further develop a lifelong learning method to automatically improve the model's capability from interactions with dynamic environments. The experiments conducted in both the RoboMimic simulation and real-world scenarios prove the superior generalization and robustness of our framework across a variety of manipulation tasks. Our code is released at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 建立可泛化的自我修正系统对于机器人从故障中恢复至关重要。尽管多模态大型语言模型（MLLMs）的发展赋予了机器人语义反思能力以应对故障，但将语义反思转化为如何纠正细粒度机器人动作仍是一个重大挑战。为了解决这一差距，我们构建了Phoenix框架，该框架利用运动指令作为桥梁，连接高层语义反思与低层机器人动作纠正。在基于运动的自我反思框架中，我们从使用MLLMs的双重过程运动调整机制开始，将语义反思转化为粗粒度运动指令调整。为了利用该运动指令引导如何纠正细粒度机器人动作，我们提出了一个多任务运动条件化扩散策略，结合视觉观察进行高频率的机器人动作纠正。通过结合这两种模型，我们将对泛化能力的需求从低层级操作策略转移至由MLLMs驱动的运动调整模型，并促进了精确、细粒度的机器人动作纠正。利用该框架，我们进一步开发了一种终身学习方法，可自动通过与动态环境的交互来提高模型能力。我们在RoboMimic仿真和真实世界场景中的实验证明了该框架在各种操作任务中具有优越的泛化能力和鲁棒性。我们的代码发布在\href{this https URL}{this https URL}。 

---
# Modality Selection and Skill Segmentation via Cross-Modality Attention 

**Title (ZH)**: 跨模态注意力驱动的模态选择与技能分割 

**Authors**: Jiawei Jiang, Kei Ota, Devesh K. Jha, Asako Kanezaki  

**Link**: [PDF](https://arxiv.org/pdf/2504.14573)  

**Abstract**: Incorporating additional sensory modalities such as tactile and audio into foundational robotic models poses significant challenges due to the curse of dimensionality. This work addresses this issue through modality selection. We propose a cross-modality attention (CMA) mechanism to identify and selectively utilize the modalities that are most informative for action generation at each timestep. Furthermore, we extend the application of CMA to segment primitive skills from expert demonstrations and leverage this segmentation to train a hierarchical policy capable of solving long-horizon, contact-rich manipulation tasks. 

**Abstract (ZH)**: 将触觉和音频等额外的感官模态融入基础机器人模型中面临着维度灾难的问题。本工作通过模态选择来应对这一问题。我们提出了一种跨模态注意力（CMA）机制，以识别并选择性利用在每个时间步对未来动作生成最有信息量的模态。此外，我们将CMA的应用扩展到从专家演示中分割原始技能，并利用这种分割来训练一个分层策略，以解决长期 horizon、接触丰富的操作任务。 

---
# NoWag: A Unified Framework for Shape Preserving Compression of Large Language Models 

**Title (ZH)**: NoWag：大规模语言模型形状保持压缩的统一框架 

**Authors**: Lawrence Liu, Inesh Chakrabarti, Yixiao Li, Mengdi Wang, Tuo Zhao, Lin F. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14569)  

**Abstract**: Large language models (LLMs) exhibit remarkable performance across various natural language processing tasks but suffer from immense computational and memory demands, limiting their deployment in resource-constrained environments. To address this challenge, we propose NoWag: (Normalized Weight and Activation Guided Compression), a unified framework for zero-shot shape preserving compression algorithms. We compressed Llama-2 7B/13B/70B and Llama-3 8/70BB models, using two popular forms of shape-preserving compression, vector quantization NoWag-VQ (NoWag for Vector Quantization), and unstructured/semi-structured pruning NoWag-P (NoWag for Pruning). We found that NoWag-VQ significantly outperforms state-of-the-art zero shot VQ, and that NoWag-P performs competitively against state-of-the-art methods. These results suggest commonalities between these compression paradigms that could inspire future work. Our code is available at this https URL 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各类自然语言处理任务中表现出色，但在计算和内存需求方面面临巨大挑战，限制了其在资源受限环境中的部署。为解决这一挑战，我们提出了NoWag：（归一化权重和激活引导压缩）统一框架，用于零样本形状保留压缩算法。我们使用向量量化NoWag-VQ（NoWag用于向量量化）和非结构化/半结构化剪枝NoWag-P（NoWag用于剪枝）对Llama-2 7B/13B/70B和Llama-3 8/70BB模型进行了压缩。结果表明，NoWag-VQ 显著优于当前最先进的零样本向量量化方法，NoWag-P 在性能上与最先进的方法竞争。这些结果表明这些压缩范式的共性可能会启发未来的工作。我们的代码可在以下链接获取：这个 https URL 

---
# ReasoningV: Efficient Verilog Code Generation with Adaptive Hybrid Reasoning Model 

**Title (ZH)**: ReasoningV：自适应混合推理模型驱动的高效Verilog代码生成 

**Authors**: Haiyan Qin, Zhiwei Xie, Jingjing Li, Liangchen Li, Xiaotong Feng, Junzhan Liu, Wang Kang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14560)  

**Abstract**: Large Language Models (LLMs) have advanced Verilog code generation significantly, yet face challenges in data quality, reasoning capabilities, and computational efficiency. This paper presents ReasoningV, a novel model employing a hybrid reasoning strategy that integrates trained intrinsic capabilities with dynamic inference adaptation for Verilog code generation. Our framework introduces three complementary innovations: (1) ReasoningV-5K, a high-quality dataset of 5,000 functionally verified instances with reasoning paths created through multi-dimensional filtering of PyraNet samples; (2) a two-stage training approach combining parameter-efficient fine-tuning for foundational knowledge with full-parameter optimization for enhanced reasoning; and (3) an adaptive reasoning mechanism that dynamically adjusts reasoning depth based on problem complexity, reducing token consumption by up to 75\% while preserving performance. Experimental results demonstrate ReasoningV's effectiveness with a pass@1 accuracy of 57.8\% on VerilogEval-human, achieving performance competitive with leading commercial models like Gemini-2.0-flash (59.5\%) and exceeding the previous best open-source model by 10.4 percentage points. ReasoningV offers a more reliable and accessible pathway for advancing AI-driven hardware design automation, with our model, data, and code available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）显著提升了Verilog代码生成，但仍面临数据质量、推理能力和计算效率的挑战。本文提出ReasoningV，这是一种采用混合推理策略的新模型，结合了训练内在能力与动态推理适应性以生成Verilog代码。我们的框架引入了三项互补创新：（1）ReasoningV-5K，一个包含5,000个功能验证实例的高质量数据集，这些实例通过多维度筛选PyraNet样本生成推理路径；（2）一种两阶段训练方法，结合参数高效微调基础知识与全参数优化以增强推理能力；（3）一种适应性推理机制，根据问题复杂度动态调整推理深度，最多减少75%的令牌消耗，同时保持性能。实验结果表明，ReasoningV在VerilogEval-human上的通过率@1为57.8%，性能与Gemini-2.0-flash（59.5%）等领先商用模型相当，且超越最佳开源模型10.4个百分点。ReasoningV为推进AI驱动的硬件设计自动化提供了一条更加可靠和可访问的途径，我们的模型、数据和代码可在以下链接获取：this https URL。 

---
# VGNC: Reducing the Overfitting of Sparse-view 3DGS via Validation-guided Gaussian Number Control 

**Title (ZH)**: VGNC: 通过验证引导的高斯数字控制减少稀视角3DGS过拟合 

**Authors**: Lifeng Lin, Rongfeng Lu, Quan Chen, Haofan Ren, Ming Lu, Yaoqi Sun, Chenggang Yan, Anke Xue  

**Link**: [PDF](https://arxiv.org/pdf/2504.14548)  

**Abstract**: Sparse-view 3D reconstruction is a fundamental yet challenging task in practical 3D reconstruction applications. Recently, many methods based on the 3D Gaussian Splatting (3DGS) framework have been proposed to address sparse-view 3D reconstruction. Although these methods have made considerable advancements, they still show significant issues with overfitting. To reduce the overfitting, we introduce VGNC, a novel Validation-guided Gaussian Number Control (VGNC) approach based on generative novel view synthesis (NVS) models. To the best of our knowledge, this is the first attempt to alleviate the overfitting issue of sparse-view 3DGS with generative validation images. Specifically, we first introduce a validation image generation method based on a generative NVS model. We then propose a Gaussian number control strategy that utilizes generated validation images to determine the optimal Gaussian numbers, thereby reducing the issue of overfitting. We conducted detailed experiments on various sparse-view 3DGS baselines and datasets to evaluate the effectiveness of VGNC. Extensive experiments show that our approach not only reduces overfitting but also improves rendering quality on the test set while decreasing the number of Gaussian points. This reduction lowers storage demands and accelerates both training and rendering. The code will be released. 

**Abstract (ZH)**: 基于验证引导的高斯点数控制（VGNC）：生成式新视角合成在稀疏视角3D重建中的应用 

---
# Causality for Natural Language Processing 

**Title (ZH)**: 自然语言处理中的因果关系 

**Authors**: Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2504.14530)  

**Abstract**: Causal reasoning is a cornerstone of human intelligence and a critical capability for artificial systems aiming to achieve advanced understanding and decision-making. This thesis delves into various dimensions of causal reasoning and understanding in large language models (LLMs). It encompasses a series of studies that explore the causal inference skills of LLMs, the mechanisms behind their performance, and the implications of causal and anticausal learning for natural language processing (NLP) tasks. Additionally, it investigates the application of causal reasoning in text-based computational social science, specifically focusing on political decision-making and the evaluation of scientific impact through citations. Through novel datasets, benchmark tasks, and methodological frameworks, this work identifies key challenges and opportunities to improve the causal capabilities of LLMs, providing a comprehensive foundation for future research in this evolving field. 

**Abstract (ZH)**: 因果推理是人类智能的基础，也是旨在实现高级理解和决策的人工系统的关键能力。本论文深入探讨了大型语言模型（LLMs）在因果推理和理解方面的各种维度。它涉及一系列研究，探索LLMs的因果推理技能、其性能背后的机制以及因果学习和反因果学习对自然语言处理（NLP）任务的影响。此外，该研究调查了因果推理在基于文本的计算社会科学中的应用，特别关注于政治决策和通过引用评估科学影响。通过引入新型数据集、基准任务和方法论框架，本研究识别了改进LLMs因果能力的关键挑战和机遇，为这一 rapidly发展的领域提供了全面的研究基础。 

---
# Biased by Design: Leveraging AI Biases to Enhance Critical Thinking of News Readers 

**Title (ZH)**: 设计偏见：利用AI偏见以增强新闻读者的批判性思维 

**Authors**: Liudmila Zavolokina, Kilian Sprenkamp, Zoya Katashinskaya, Daniel Gordon Jones  

**Link**: [PDF](https://arxiv.org/pdf/2504.14522)  

**Abstract**: This paper explores the design of a propaganda detection tool using Large Language Models (LLMs). Acknowledging the inherent biases in AI models, especially in political contexts, we investigate how these biases might be leveraged to enhance critical thinking in news consumption. Countering the typical view of AI biases as detrimental, our research proposes strategies of user choice and personalization in response to a user's political stance, applying psychological concepts of confirmation bias and cognitive dissonance. We present findings from a qualitative user study, offering insights and design recommendations (bias awareness, personalization and choice, and gradual introduction of diverse perspectives) for AI tools in propaganda detection. 

**Abstract (ZH)**: 本研究探讨了使用大型语言模型（LLMs）设计宣传检测工具的方法。考虑到AI模型内在的偏见，尤其是在政治语境中，我们考察了这些偏见如何被利用以增强新闻消费中的批判性思维。不同于将AI偏见视为有害的看法，我们的研究提出了一种针对用户政治立场的用户选择和个性化策略，应用了确证偏见和认知不协调的心理学概念。我们呈现了定性用户研究的结果，为宣传检测中的AI工具提供了关于偏见意识、个性化和逐渐引入多样化视角的设计建议。 

---
# SlimPipe: Memory-Thrifty and Efficient Pipeline Parallelism for Long-Context LLM Training 

**Title (ZH)**: SlimPipe：节省内存且高效的长上下文LLM训练管道并行ism 

**Authors**: Zhouyang Li, Yuliang Liu, Wei Zhang, Tailing Yuan, Bin Chen, Chengru Song, Di Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14519)  

**Abstract**: Pipeline Parallelism (PP) serves as a crucial technique for training Large Language Models (LLMs), owing to its capability to alleviate memory pressure from model states with relatively low communication overhead. However, in long-context scenarios, existing pipeline parallelism methods fail to address the substantial activation memory pressure, primarily due to the peak memory consumption resulting from the accumulation of activations across multiple microbatches. Moreover, these approaches inevitably introduce considerable pipeline bubbles, further hindering efficiency.
To tackle these challenges, we propose SlimPipe, a novel approach to fine-grained pipeline parallelism that employs uniform sequence slicing coupled with one-forward-one-backward (1F1B) schedule. It reduces the accumulated activations from several microbatches to just one, which is split into several slices. Although the slices are evenly partitioned, the computation cost is not equal across slices due to causal attention. We develop a sophisticated workload redistribution technique to address this load imbalance. SlimPipe achieves (1) near-zero memory overhead and (2) minimal pipeline bubbles simultaneously. The effectiveness of SlimPipe has been proven by thorough testing with diverse model architectures, context window sizes, and SlimPipe-specific configurations. For example, on the Llama 70B model, compared to state-of-the-art methods, SlimPipe significantly boosts the Model FLOPs Utilization (MFU) to up to $1.57\times$ for a context length of 512K. More notably, for a context length of 2048K, it maintains over 45% utilization on 256 NVIDIA Hopper 80GB GPUs, while other approaches either suffer significant performance drops or fail entirely due to memory constraints. 

**Abstract (ZH)**: SlimPipe：一种新颖的精细粒度管道并行方法 

---
# On Dimension-Free Transformer: An Application of STP to AI 

**Title (ZH)**: 维度无关的变压器：STP在AI中的应用 

**Authors**: Daizhan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.14514)  

**Abstract**: The matrix expressions for every parts of a transformer are firstly described. Based on semi-tensor product (STP) of matrices the hypervectors are reconsidered and the linear transformation over hypervectors is constructed by using projection. Its properties and calculating formulas are obtained. Using projection-based transformation of hypervector (PBTH), the framework of dimension-free transformer (DFT) is proposed by verifying each linear transformation in a transformer and replacing it by a proper PBTH, which allows the inputs and outputs being of arbitrary dimensions. Using balanced information about all entries, DFT must be more efficient in dealing with signals. 

**Abstract (ZH)**: 变压器中每一部分的矩阵表达式首先被描述。基于矩阵的半张量积（STP），重新考虑了超向量，并通过投影构建了超向量的线性变换，获得了其性质和计算公式。利用基于投影的超向量变换（PBTH），通过验证变压器中的每一线性变换并用适当的PBTH替换，提出了维度无关变压器（DFT）的框架，使得输入和输出可以具有任意维度。利用所有条目平衡的信息，DFT在处理信号时必然更高效。 

---
# DreamID: High-Fidelity and Fast diffusion-based Face Swapping via Triplet ID Group Learning 

**Title (ZH)**: DreamID: 基于三重ID组学习的高保真快速人脸互换 

**Authors**: Fulong Ye, Miao Hua, Pengze Zhang, Xinghui Li, Qichao Sun, Songtao Zhao, Qian He, Xinglong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14509)  

**Abstract**: In this paper, we introduce DreamID, a diffusion-based face swapping model that achieves high levels of ID similarity, attribute preservation, image fidelity, and fast inference speed. Unlike the typical face swapping training process, which often relies on implicit supervision and struggles to achieve satisfactory results. DreamID establishes explicit supervision for face swapping by constructing Triplet ID Group data, significantly enhancing identity similarity and attribute preservation. The iterative nature of diffusion models poses challenges for utilizing efficient image-space loss functions, as performing time-consuming multi-step sampling to obtain the generated image during training is impractical. To address this issue, we leverage the accelerated diffusion model SD Turbo, reducing the inference steps to a single iteration, enabling efficient pixel-level end-to-end training with explicit Triplet ID Group supervision. Additionally, we propose an improved diffusion-based model architecture comprising SwapNet, FaceNet, and ID Adapter. This robust architecture fully unlocks the power of the Triplet ID Group explicit supervision. Finally, to further extend our method, we explicitly modify the Triplet ID Group data during training to fine-tune and preserve specific attributes, such as glasses and face shape. Extensive experiments demonstrate that DreamID outperforms state-of-the-art methods in terms of identity similarity, pose and expression preservation, and image fidelity. Overall, DreamID achieves high-quality face swapping results at 512*512 resolution in just 0.6 seconds and performs exceptionally well in challenging scenarios such as complex lighting, large angles, and occlusions. 

**Abstract (ZH)**: 基于扩散模型的DreamID面部替换模型：高身份相似度、属性保真度、图像 fidelity 和快速推断速度 

---
# LBM-GNN: Graph Neural Network Enhanced Lattice Boltzmann Method 

**Title (ZH)**: LBM-GNN： lattice Boltzmann 方法增强的图神经网络 

**Authors**: Yue Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.14494)  

**Abstract**: In this paper, we present LBM-GNN, a novel approach that enhances the traditional Lattice Boltzmann Method (LBM) with Graph Neural Networks (GNNs). We apply this method to fluid dynamics simulations, demonstrating improved stability and accuracy compared to standard LBM implementations. The method is validated using benchmark problems such as the Taylor-Green vortex, focusing on accuracy, conservation properties, and performance across different Reynolds numbers and grid resolutions. Our results indicate that GNN-enhanced LBM can maintain better conservation properties while improving numerical stability at higher Reynolds numbers. 

**Abstract (ZH)**: LBM-GNN：通过图神经网络增强的晶格玻尔兹曼方法及其在流体动力学模拟中的应用 

---
# FinSage: A Multi-aspect RAG System for Financial Filings Question Answering 

**Title (ZH)**: FinSage: 一种多方面语料库检索系统用于财务报表问答 

**Authors**: Xinyu Wang, Jijun Chi, Zhenghan Tai, Tung Sum Thomas Kwok, Muzhi Li, Zhuhong Li, Hailin He, Yuchen Hua, Peng Lu, Suyuchen Wang, Yihong Wu, Jerry Huang, Ling Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.14493)  

**Abstract**: Leveraging large language models in real-world settings often entails a need to utilize domain-specific data and tools in order to follow the complex regulations that need to be followed for acceptable use. Within financial sectors, modern enterprises increasingly rely on Retrieval-Augmented Generation (RAG) systems to address complex compliance requirements in financial document workflows. However, existing solutions struggle to account for the inherent heterogeneity of data (e.g., text, tables, diagrams) and evolving nature of regulatory standards used in financial filings, leading to compromised accuracy in critical information extraction. We propose the FinSage framework as a solution, utilizing a multi-aspect RAG framework tailored for regulatory compliance analysis in multi-modal financial documents. FinSage introduces three innovative components: (1) a multi-modal pre-processing pipeline that unifies diverse data formats and generates chunk-level metadata summaries, (2) a multi-path sparse-dense retrieval system augmented with query expansion (HyDE) and metadata-aware semantic search, and (3) a domain-specialized re-ranking module fine-tuned via Direct Preference Optimization (DPO) to prioritize compliance-critical content. Extensive experiments demonstrate that FinSage achieves an impressive recall of 92.51% on 75 expert-curated questions derived from surpasses the best baseline method on the FinanceBench question answering datasets by 24.06% in accuracy. Moreover, FinSage has been successfully deployed as financial question-answering agent in online meetings, where it has already served more than 1,200 people. 

**Abstract (ZH)**: 利用大规模语言模型在实际应用场景中往往需要使用领域特定的数据和工具以遵循复杂的合规要求。在金融领域，现代企业越来越多地依赖检索增强生成（RAG）系统来解决金融文档流程中的复杂合规要求。然而，现有的解决方案难以应对数据的内在异质性（例如，文本、表格、图表）和监管标准的不断发展变化，这导致关键信息提取的准确性受到影响。我们提出FinSage框架作为解决方案，利用一个针对多模态金融文件中合规分析的多方面RAG框架。FinSage引入了三个创新组件：（1）一个多模态预处理流水线，统一多种数据格式并生成片段级元数据摘要；（2）一个增强查询扩展（HyDE）和元数据意识语义搜索的多路径稀疏密集检索系统；（3）一个通过直接偏好优化（DPO）微调的领域专用重排模块，优先处理合规关键内容。广泛实验表明，FinSage在75个专家策划的问题上实现了92.51%的召回率，在FinanceBench问答数据集上比最佳基线方法的准确性高出24.06%。此外，FinSage已被成功部署为在线会议中的金融问答代理，已经为超过1,200人提供了服务。 

---
# ParaPO: Aligning Language Models to Reduce Verbatim Reproduction of Pre-training Data 

**Title (ZH)**: ParaPO: 减少预训练数据直搬用的语言模型对齐方法 

**Authors**: Tong Chen, Faeze Brahman, Jiacheng Liu, Niloofar Mireshghallah, Weijia Shi, Pang Wei Koh, Luke Zettlemoyer, Hannaneh Hajishirzi  

**Link**: [PDF](https://arxiv.org/pdf/2504.14452)  

**Abstract**: Language models (LMs) can memorize and reproduce segments from their pretraining data verbatim even in non-adversarial settings, raising concerns about copyright, plagiarism, privacy, and creativity. We introduce Paraphrase Preference Optimization (ParaPO), a post-training method that fine-tunes LMs to reduce unintentional regurgitation while preserving their overall utility. ParaPO trains LMs to prefer paraphrased versions of memorized segments over the original verbatim content from the pretraining data. To maintain the ability to recall famous quotations when appropriate, we develop a variant of ParaPO that uses system prompts to control regurgitation behavior. In our evaluation on Llama3.1-8B, ParaPO consistently reduces regurgitation across all tested datasets (e.g., reducing the regurgitation metric from 17.3 to 12.9 in creative writing), whereas unlearning methods used in prior work to mitigate regurgitation are less effective outside their targeted unlearned domain (from 17.3 to 16.9). When applied to the instruction-tuned Tulu3-8B model, ParaPO with system prompting successfully preserves famous quotation recall while reducing unintentional regurgitation (from 8.7 to 6.3 in creative writing) when prompted not to regurgitate. In contrast, without ParaPO tuning, prompting the model not to regurgitate produces only a marginal reduction (8.7 to 8.4). 

**Abstract (ZH)**: Paraphrase Preference Optimization: A Post-Training Method to Reduce Unintentional Regurgitation While Preserving Overall Utility 

---
# LoRe: Personalizing LLMs via Low-Rank Reward Modeling 

**Title (ZH)**: LoRe: 通过低秩奖励建模个性化LLMs 

**Authors**: Avinandan Bose, Zhihan Xiong, Yuejie Chi, Simon Shaolei Du, Lin Xiao, Maryam Fazel  

**Link**: [PDF](https://arxiv.org/pdf/2504.14439)  

**Abstract**: Personalizing large language models (LLMs) to accommodate diverse user preferences is essential for enhancing alignment and user satisfaction. Traditional reinforcement learning from human feedback (RLHF) approaches often rely on monolithic value representations, limiting their ability to adapt to individual preferences. We introduce a novel framework that leverages low-rank preference modeling to efficiently learn and generalize user-specific reward functions. By representing reward functions in a low-dimensional subspace and modeling individual preferences as weighted combinations of shared basis functions, our approach avoids rigid user categorization while enabling scalability and few-shot adaptation. We validate our method on multiple preference datasets, demonstrating superior generalization to unseen users and improved accuracy in preference prediction tasks. 

**Abstract (ZH)**: 个性化大型语言模型以适应多样化用户偏好对于增强对齐和用户满意度至关重要。传统的基于人类反馈的强化学习（RLHF）方法往往依赖于单一的价值表示，限制了其适应个体偏好的能力。我们提出了一种新的框架，利用低秩偏好建模来高效学习和泛化用户特定的奖励函数。通过在低维子空间中表示奖励函数，并将个体偏好建模为共享基函数的加权组合，我们的方法避免了僵硬的用户分类，同时实现了可扩展性和少量示例的适应性。我们在多个偏好数据集上验证了该方法，展示了对未见用户的优越泛化能力和在偏好预测任务中的改进准确性。 

---
# ResNetVLLM -- Multi-modal Vision LLM for the Video Understanding Task 

**Title (ZH)**: ResNetVLLM -- 多模态视觉LLM在视频理解任务中的应用 

**Authors**: Ahmad Khalil, Mahmoud Khalil, Alioune Ngom  

**Link**: [PDF](https://arxiv.org/pdf/2504.14432)  

**Abstract**: In this paper, we introduce ResNetVLLM (ResNet Vision LLM), a novel cross-modal framework for zero-shot video understanding that integrates a ResNet-based visual encoder with a Large Language Model (LLM. ResNetVLLM addresses the challenges associated with zero-shot video models by avoiding reliance on pre-trained video understanding models and instead employing a non-pretrained ResNet to extract visual features. This design ensures the model learns visual and semantic representations within a unified architecture, enhancing its ability to generate accurate and contextually relevant textual descriptions from video inputs. Our experimental results demonstrate that ResNetVLLM achieves state-of-the-art performance in zero-shot video understanding (ZSVU) on several benchmarks, including MSRVTT-QA, MSVD-QA, TGIF-QA FrameQA, and ActivityNet-QA. 

**Abstract (ZH)**: ResNetVLLM：一种基于ResNet的视觉编码器与大型语言模型结合的零样本视频理解新型跨模态框架 

---
# ResNetVLLM-2: Addressing ResNetVLLM's Multi-Modal Hallucinations 

**Title (ZH)**: ResNetVLLM-2: 解决ResNetVLLM的多模态幻觉问题 

**Authors**: Ahmad Khalil, Mahmoud Khalil, Alioune Ngom  

**Link**: [PDF](https://arxiv.org/pdf/2504.14429)  

**Abstract**: Large Language Models (LLMs) have transformed natural language processing (NLP) tasks, but they suffer from hallucination, generating plausible yet factually incorrect content. This issue extends to Video-Language Models (VideoLLMs), where textual descriptions may inaccurately represent visual content, resulting in multi-modal hallucinations. In this paper, we address hallucination in ResNetVLLM, a video-language model combining ResNet visual encoders with LLMs. We introduce a two-step protocol: (1) a faithfulness detection strategy that uses a modified Lynx model to assess semantic alignment between generated captions and ground-truth video references, and (2) a hallucination mitigation strategy using Retrieval-Augmented Generation (RAG) with an ad-hoc knowledge base dynamically constructed during inference. Our enhanced model, ResNetVLLM-2, reduces multi-modal hallucinations by cross-verifying generated content against external knowledge, improving factual consistency. Evaluation on the ActivityNet-QA benchmark demonstrates a substantial accuracy increase from 54.8% to 65.3%, highlighting the effectiveness of our hallucination detection and mitigation strategies in enhancing video-language model reliability. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经重塑了自然语言处理（NLP）任务，但它们存在幻觉问题，即生成看似合理但实际上不正确的内容。这一问题也延伸到了视频语言模型（VideoLLMs），其中文本描述可能无法准确代表视觉内容，导致多模态幻觉。在本文中，我们针对结合了ResNet视觉编码器和LLMs的ResNetVLLM视频语言模型中的幻觉问题进行研究。我们提出了一种两步协议：（1）公平性检测策略，利用修改后的Lynx模型评估生成的字幕与真实视频参考之间的语义对齐；（2）使用结合临时知识库的检索增强生成（RAG）的幻觉缓解策略。通过增强模型ResNetVLLM-2，我们在生成内容与外部知识交叉验证的过程中减少了多模态幻觉，提高了事实一致性。在ActivityNet-QA基准测试上的评估显示，准确率从54.8%提升到65.3%，突显了我们提出的幻觉检测和缓解策略在提升视频语言模型可靠性方面的有效性。 

---
# Optimizing SIA Development: A Case Study in User-Centered Design for Estuary, a Multimodal Socially Interactive Agent Framework 

**Title (ZH)**: 优化SIA发展：一个基于用户中心设计的案例研究——以Estuary多模态社会交互代理框架为例 

**Authors**: Spencer Lin, Miru Jun, Basem Rizk, Karen Shieh, Scott Fisher, Sharon Mozgai  

**Link**: [PDF](https://arxiv.org/pdf/2504.14427)  

**Abstract**: This case study presents our user-centered design model for Socially Intelligent Agent (SIA) development frameworks through our experience developing Estuary, an open source multimodal framework for building low-latency real-time socially interactive agents. We leverage the Rapid Assessment Process (RAP) to collect the thoughts of leading researchers in the field of SIAs regarding the current state of the art for SIA development as well as their evaluation of how well Estuary may potentially address current research gaps. We achieve this through a series of end-user interviews conducted by a fellow researcher in the community. We hope that the findings of our work will not only assist the continued development of Estuary but also guide the development of other future frameworks and technologies for SIAs. 

**Abstract (ZH)**: 本案例研究通过我们在开发开源多模态框架Estuary（用于构建低延迟实时社交互动代理）过程中积累的经验，提出了以用户为中心的设计模型，用于社交智能代理（SIA）开发框架。我们利用快速评估过程（RAP）收集领域内领先研究人员关于SIA开发的当前技术水平及其评估，探讨Estuary如何潜在地弥补当前的研究空白。我们通过社区中另一位研究人员进行的一系列最终用户访谈实现了这一点。我们希望本工作的发现不仅能够促进Estuary的持续开发，还能够指导其他未来SIA框架和技术的发展。 

---
# Adversarial Attack for RGB-Event based Visual Object Tracking 

**Title (ZH)**: 基于RGB-事件的视觉目标跟踪的对抗攻击 

**Authors**: Qiang Chen, Xiao Wang, Haowen Wang, Bo Jiang, Lin Zhu, Dawei Zhang, Yonghong Tian, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14423)  

**Abstract**: Visual object tracking is a crucial research topic in the fields of computer vision and multi-modal fusion. Among various approaches, robust visual tracking that combines RGB frames with Event streams has attracted increasing attention from researchers. While striving for high accuracy and efficiency in tracking, it is also important to explore how to effectively conduct adversarial attacks and defenses on RGB-Event stream tracking algorithms, yet research in this area remains relatively scarce. To bridge this gap, in this paper, we propose a cross-modal adversarial attack algorithm for RGB-Event visual tracking. Because of the diverse representations of Event streams, and given that Event voxels and frames are more commonly used, this paper will focus on these two representations for an in-depth study. Specifically, for the RGB-Event voxel, we first optimize the perturbation by adversarial loss to generate RGB frame adversarial examples. For discrete Event voxel representations, we propose a two-step attack strategy, more in detail, we first inject Event voxels into the target region as initialized adversarial examples, then, conduct a gradient-guided optimization by perturbing the spatial location of the Event voxels. For the RGB-Event frame based tracking, we optimize the cross-modal universal perturbation by integrating the gradient information from multimodal data. We evaluate the proposed approach against attacks on three widely used RGB-Event Tracking datasets, i.e., COESOT, FE108, and VisEvent. Extensive experiments show that our method significantly reduces the performance of the tracker across numerous datasets in both unimodal and multimodal scenarios. The source code will be released on this https URL 

**Abstract (ZH)**: 跨模态的RGB-事件流视觉跟踪对抗攻击算法 

---
# Planet as a Brain: Towards Internet of AgentSites based on AIOS Server 

**Title (ZH)**: 行星作为大脑：基于AIOS服务器的代理站点互联网owards Internet of AgentSites based on AIOS Server 

**Authors**: Xiang Zhang, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14411)  

**Abstract**: The internet is undergoing a historical transformation from the "Internet of Websites" to the "Internet of AgentSites." While traditional Websites served as the foundation for information hosting and dissemination, a new frontier is emerging where AgentSites serve as the hubs of the internet, where each AgentSite hosts one or more AI agents that receive tasks, address them, and deliver actionable solutions, marking a significant shift in the digital landscape and representing the next generation of online ecosystems. Under this vision, AIOS, the AI Agent Operating System, serves as the server for the development, deployment and execution of AI agents, which is a fundamental infrastructure for the Internet of Agentsites.
In this paper, we introduce AIOS Server, a runtime framework to host agents and enable global-scale collaboration among decentralized agents. AIOS Server provides a communication protocol leveraging the Model Context Protocol (MCP) and JSON-RPC to enable agent-agent or human-agent interactions. Each AIOS node operates as a server to host and execute agents, while supporting peer-to-peer coordination without reliance on centralized orchestration. Based on AIOS Server, we further present the world's first practically deployed Internet of Agentsites (AIOS-IoA), including AgentHub for agent registration and discovery and AgentChat for interactive communication, at this https URL. The agent discovery mechanism based on Distributed Hash Tables (DHT) and a Gossip protocol serves as the search engine for the internet of agentsites. This work provides a practical foundation for building the Internet of Agentsites-a new paradigm where autonomous agents become first-class citizens of the web. The implementation is available at this https URL and will be integrated into the AIOS main branch at this https URL. 

**Abstract (ZH)**: 互联网正在经历从“网站网”到“智能站点网”的历史转变。传统网站作为信息托管和传播的基础，一个新的前沿领域正逐渐兴起，智能站点成为互联网的枢纽，每个智能站点托管一个或多个AI代理，接收任务、解决任务并提供可执行的解决方案，标志着数字景观的重大转变，并代表着下一代在线生态系统。在此愿景下，AIOS（AI代理操作系统）作为开发、部署和执行AI代理的服务器，是智能站点网的基础基础设施。

在本文中，我们介绍了AIOS Server，这是一个运行时框架，用于托管代理并促进去中心化代理的全球规模协作。AIOS Server利用模型上下文协议（MCP）和JSON-RPC提供通信协议，以支持代理与代理或人与代理之间的交互。每个AIOS节点作为一个服务器来托管和执行代理，并支持去中心化的协调机制而无需依赖集中式编排。基于AIOS Server，我们进一步介绍了第一个实际部署的智能站点网（AIOS-IoA），包括AgentHub（智能站点注册和发现）和AgentChat（互动通信），详情请参见此https://链接。基于分布式哈希表（DHT）和Gossip协议的智能站点发现机制作为智能站点网的搜索引擎。本项工作为构建智能站点网提供了实际的基础，使自治代理成为网络中的头等公民。相关实现可在此https://链接获取，并将集成到AIOS主分支中。 

---
# Data Augmentation Using Neural Acoustic Fields With Retrieval-Augmented Pre-training 

**Title (ZH)**: 基于检索增强预训练的神经声学场数据增强方法 

**Authors**: Christopher Ick, Gordon Wichern, Yoshiki Masuyama, François G. Germain, Jonathan Le Roux  

**Link**: [PDF](https://arxiv.org/pdf/2504.14409)  

**Abstract**: This report details MERL's system for room impulse response (RIR) estimation submitted to the Generative Data Augmentation Workshop at ICASSP 2025 for Augmenting RIR Data (Task 1) and Improving Speaker Distance Estimation (Task 2). We first pre-train a neural acoustic field conditioned by room geometry on an external large-scale dataset in which pairs of RIRs and the geometries are provided. The neural acoustic field is then adapted to each target room by using the enrollment data, where we leverage either the provided room geometries or geometries retrieved from the external dataset, depending on availability. Lastly, we predict the RIRs for each pair of source and receiver locations specified by Task 1, and use these RIRs to train the speaker distance estimation model in Task 2. 

**Abstract (ZH)**: MERL的房间冲激响应估计系统：用于ICASSP 2025生成数据增强工作坊的房间冲激响应数据增强（任务1）和演讲者距离估计改进（任务2）。 

---
# ScholarMate: A Mixed-Initiative Tool for Qualitative Knowledge Work and Information Sensemaking 

**Title (ZH)**: ScholarMate: 一种混合主动型工具，用于定性知识工作和信息意义构建 

**Authors**: Runlong Ye, Patrick Yung Kang Lee, Matthew Varona, Oliver Huang, Carolina Nobre  

**Link**: [PDF](https://arxiv.org/pdf/2504.14406)  

**Abstract**: Synthesizing knowledge from large document collections is a critical yet increasingly complex aspect of qualitative research and knowledge work. While AI offers automation potential, effectively integrating it into human-centric sensemaking workflows remains challenging. We present ScholarMate, an interactive system designed to augment qualitative analysis by unifying AI assistance with human oversight. ScholarMate enables researchers to dynamically arrange and interact with text snippets on a non-linear canvas, leveraging AI for theme suggestions, multi-level summarization, and contextual naming, while ensuring transparency through traceability to source documents. Initial pilot studies indicated that users value this mixed-initiative approach, finding the balance between AI suggestions and direct manipulation crucial for maintaining interpretability and trust. We further demonstrate the system's capability through a case study analyzing 24 papers. By balancing automation with human control, ScholarMate enhances efficiency and supports interpretability, offering a valuable approach for productive human-AI collaboration in demanding sensemaking tasks common in knowledge work. 

**Abstract (ZH)**: 从大规模文档集合中合成知识是定性研究和知识工作中一个关键但日益复杂的方面。尽管人工智能提供了自动化潜力，将其有效集成到以人文为中心的意义建构工作流程中仍然具有挑战性。我们介绍了ScholarMate，这是一个交互系统，旨在通过统一人工智能辅助和人类监督来增强定性分析。ScholarMate使研究者能够动态地在非线性画布上排列和交互文本片段，利用人工智能进行主题建议、多级总结和上下文命名，同时通过溯源保持透明度。初步的试点研究显示，用户欣赏这种混合主动的方法，认为在人工智能建议和直接操作之间找到平衡对保持解释性和信任至关重要。我们通过分析24篇论文的案例研究进一步展示了该系统的功能。通过平衡自动化和人类控制，ScholarMate提高了效率并支持了解释性，为知识工作中常见的意义建构任务提供了有价值的人机协作方法。 

---
# Hydra: An Agentic Reasoning Approach for Enhancing Adversarial Robustness and Mitigating Hallucinations in Vision-Language Models 

**Title (ZH)**: Hydra: 一种增强视觉-语言模型对抗鲁棒性并缓解幻觉的机构性推理方法 

**Authors**: Chung-En, Hsuan-Chih, Chen, Brian Jalaian, Nathaniel D. Bastian  

**Link**: [PDF](https://arxiv.org/pdf/2504.14395)  

**Abstract**: To develop trustworthy Vision-Language Models (VLMs), it is essential to address adversarial robustness and hallucination mitigation, both of which impact factual accuracy in high-stakes applications such as defense and healthcare. Existing methods primarily focus on either adversarial defense or hallucination post-hoc correction, leaving a gap in unified robustness strategies. We introduce \textbf{Hydra}, an adaptive agentic framework that enhances plug-in VLMs through iterative reasoning, structured critiques, and cross-model verification, improving both resilience to adversarial perturbations and intrinsic model errors. Hydra employs an Action-Critique Loop, where it retrieves and critiques visual information, leveraging Chain-of-Thought (CoT) and In-Context Learning (ICL) techniques to refine outputs dynamically. Unlike static post-hoc correction methods, Hydra adapts to both adversarial manipulations and intrinsic model errors, making it robust to malicious perturbations and hallucination-related inaccuracies. We evaluate Hydra on four VLMs, three hallucination benchmarks, two adversarial attack strategies, and two adversarial defense methods, assessing performance on both clean and adversarial inputs. Results show that Hydra surpasses plug-in VLMs and state-of-the-art (SOTA) dehallucination methods, even without explicit adversarial defenses, demonstrating enhanced robustness and factual consistency. By bridging adversarial resistance and hallucination mitigation, Hydra provides a scalable, training-free solution for improving the reliability of VLMs in real-world applications. 

**Abstract (ZH)**: 开发可信的视觉-语言模型 (VLMs) 需要解决对抗鲁棒性和幻觉缓解问题，这两者在高风险应用如国防和医疗保健中影响事实准确性。现有方法主要集中在对抗防御或事后幻觉修正上，留下了统一鲁棒性策略的缺口。我们引入了**Hydra**，一个自适应代理框架，通过迭代推理、结构化批判和跨模型验证增强插件VLMs，提高其对抗扰动的抗性和内在模型错误。Hydra 使用行动-批判循环，在此过程中检索和批判视觉信息，并利用链式思考（CoT）和上下文相关学习（ICL）技术动态优化输出。与静态事后修正方法不同，Hydra 能够应对对抗操作和内在模型错误，使其对恶意扰动和幻觉相关不准确具有鲁棒性。我们对四种VLMs、三种幻觉基准、两种对抗攻击策略和两种对抗防御方法进行了评估，评估其在干净和对抗输入上的性能。结果显示，Hydra 超过了插件VLMs和最新的去幻觉方法，甚至在没有明确的对抗防御措施的情况下也表现出更强的鲁棒性和事实一致性。通过结合对抗抵抗和幻觉缓解，Hydra 提供了一个可扩展的、无需训练的解决方案，以提高视觉-语言模型在实际应用中的可靠性。 

---
# LOOPE: Learnable Optimal Patch Order in Positional Embeddings for Vision Transformers 

**Title (ZH)**: LOOPE: 可学习的位置嵌入最优patches顺序在视觉变换器中的应用 

**Authors**: Md Abtahi Majeed Chowdhury, Md Rifat Ur Rahman, Akil Ahmad Taki  

**Link**: [PDF](https://arxiv.org/pdf/2504.14386)  

**Abstract**: Positional embeddings (PE) play a crucial role in Vision Transformers (ViTs) by providing spatial information otherwise lost due to the permutation invariant nature of self attention. While absolute positional embeddings (APE) have shown theoretical advantages over relative positional embeddings (RPE), particularly due to the ability of sinusoidal functions to preserve spatial inductive biases like monotonicity and shift invariance, a fundamental challenge arises when mapping a 2D grid to a 1D sequence. Existing methods have mostly overlooked or never explored the impact of patch ordering in positional embeddings. To address this, we propose LOOPE, a learnable patch-ordering method that optimizes spatial representation for a given set of frequencies, providing a principled approach to patch order optimization. Empirical results show that our PE significantly improves classification accuracy across various ViT architectures. To rigorously evaluate the effectiveness of positional embeddings, we introduce the "Three Cell Experiment", a novel benchmarking framework that assesses the ability of PEs to retain relative and absolute positional information across different ViT architectures. Unlike standard evaluations, which typically report a performance gap of 4 to 6% between models with and without PE, our method reveals a striking 30 to 35% difference, offering a more sensitive diagnostic tool to measure the efficacy of PEs. Our experimental analysis confirms that the proposed LOOPE demonstrates enhanced effectiveness in retaining both relative and absolute positional information. 

**Abstract (ZH)**: 位置嵌入（PE）在视觉变压器（ViTs）中通过提供由于自注意力的排列不变性而丢失的空间信息发挥着关键作用。虽然绝对位置嵌入（APE）在理论上展现出相对于相对位置嵌入（RPE）的优势，尤其是在保持诸如单调性和移位不变性等空间诱导偏置方面，当将2D网格映射到1D序列时，一个基本的挑战随之而来。现有的方法大多忽视或从未探索过位置嵌入中切片顺序的影响。为了解决这一问题，我们提出了一种可学习的切片顺序方法LOOPE，它针对给定的频率集优化空间表示，为切片顺序的优化提供了一种原则性的方法。我们的实验结果显示，位置嵌入显著提高了各种ViT架构的分类准确性。为了严格评估位置嵌入的有效性，我们引入了“The Three Cell Experiment”这一新的基准框架，评估位置嵌入在不同ViT架构中保留相对和绝对位置信息的能力。与标准评估相比，我们的方法揭示了高达30%到35%的显著差异，提供了更敏感的诊断工具来衡量位置嵌入的效果。我们的实验分析证实，所提出的LOOPE在保留相对和绝对位置信息方面表现出更优的效果。 

---
# Learning Enhanced Structural Representations with Block-Based Uncertainties for Ocean Floor Mapping 

**Title (ZH)**: 基于块基础不确定性学习增强的结构表示用于海底测绘 

**Authors**: Jose Marie Antonio Minoza  

**Link**: [PDF](https://arxiv.org/pdf/2504.14372)  

**Abstract**: Accurate ocean modeling and coastal hazard prediction depend on high-resolution bathymetric data; yet, current worldwide datasets are too coarse for exact numerical simulations. While recent deep learning advances have improved earth observation data resolution, existing methods struggle with the unique challenges of producing detailed ocean floor maps, especially in maintaining physical structure consistency and quantifying uncertainties. This work presents a novel uncertainty-aware mechanism using spatial blocks to efficiently capture local bathymetric complexity based on block-based conformal prediction. Using the Vector Quantized Variational Autoencoder (VQ-VAE) architecture, the integration of this uncertainty quantification framework yields spatially adaptive confidence estimates while preserving topographical features via discrete latent representations. With smaller uncertainty widths in well-characterized areas and appropriately larger bounds in areas of complex seafloor structures, the block-based design adapts uncertainty estimates to local bathymetric complexity. Compared to conventional techniques, experimental results over several ocean regions show notable increases in both reconstruction quality and uncertainty estimation reliability. This framework increases the reliability of bathymetric reconstructions by preserving structural integrity while offering spatially adaptive uncertainty estimates, so opening the path for more solid climate modeling and coastal hazard assessment. 

**Abstract (ZH)**: 高精度海底地形建模与沿海灾害预测依赖于高分辨率 bathymetric 数据；然而，当前全球数据集的分辨率尚不足以进行精确的数值模拟。尽管近年来深度学习技术提高了地球观测数据的分辨率，但现有方法在生成详细的海底地形图时面临着独特挑战，尤其是在保持物理结构一致性和量化不确定性方面。本文提出了一种新的不确定性感知机制，利用基于区块的 conforme 预测，结合区块结构有效地捕获局部海底地形的复杂性。通过采用向量量化变分自编码器（VQ-VAE）架构，该不确定性量化框架提供了空间自适应的置信度估计，同时通过离散的潜在表示保留地形特征。在特征描述良好的区域，通过减小不确定性范围；而在复杂海底结构的区域，则适当扩大不确定性范围。区块化设计使不确定性估计能够适应局部海底地形复杂性。与传统技术相比，多个海洋区域的实验结果表明，该框架在重建质量和不确定性估计可靠性方面均有显著提升。该框架通过保留结构性完整性并提供空间自适应的不确定性估计，提高了海底地形重建的可靠性，为更加坚实的气候建模和沿海灾害评估铺平了道路。 

---
# Diverse Prompts: Illuminating the Prompt Space of Large Language Models with MAP-Elites 

**Title (ZH)**: 多样的提示：利用MAP-Elites照亮大规模语言模型的提示空间 

**Authors**: Gabriel Machado Santos, Rita Maria da Silva Julia, Marcelo Zanchetta do Nascimento  

**Link**: [PDF](https://arxiv.org/pdf/2504.14367)  

**Abstract**: Prompt engineering is essential for optimizing large language models (LLMs), yet the link between prompt structures and task performance remains underexplored. This work introduces an evolutionary approach that combines context-free grammar (CFG) with the MAP-Elites algorithm to systematically explore the prompt space. Our method prioritizes quality and diversity, generating high-performing and structurally varied prompts while analyzing their alignment with diverse tasks by varying traits such as the number of examples (shots) and reasoning depth. By systematically mapping the phenotypic space, we reveal how structural variations influence LLM performance, offering actionable insights for task-specific and adaptable prompt design. Evaluated on seven BigBench Lite tasks across multiple LLMs, our results underscore the critical interplay of quality and diversity, advancing the effectiveness and versatility of LLMs. 

**Abstract (ZH)**: 基于上下文无关文法的进化方法：系统探索提示空间以优化大型语言模型的任务性能 

---
# Empirical Evaluation of Knowledge Distillation from Transformers to Subquadratic Language Models 

**Title (ZH)**: 从Transformer到亚二次语言模型的知识蒸馏 empirical evaluation 

**Authors**: Patrick Haller, Jonas Golde, Alan Akbik  

**Link**: [PDF](https://arxiv.org/pdf/2504.14366)  

**Abstract**: Knowledge distillation is a widely used technique for compressing large language models (LLMs) by training a smaller student model to mimic a larger teacher model. Typically, both the teacher and student are Transformer-based architectures, leveraging softmax attention for sequence modeling. However, the quadratic complexity of self-attention at inference time remains a significant bottleneck, motivating the exploration of subquadratic alternatives such as structured state-space models (SSMs), linear attention, and recurrent architectures. In this work, we systematically evaluate the transferability of knowledge distillation from a Transformer teacher to nine subquadratic student architectures. Our study aims to determine which subquadratic model best aligns with the teacher's learned representations and how different architectural constraints influence the distillation process. We also investigate the impact of intelligent initialization strategies, including matrix mixing and query-key-value (QKV) copying, on the adaptation process. Our empirical results on multiple NLP benchmarks provide insights into the trade-offs between efficiency and performance, highlighting key factors for successful knowledge transfer to subquadratic architectures. 

**Abstract (ZH)**: 知识蒸馏是一种广泛用于通过训练较小的学生模型来模仿较大教师模型以压缩大型语言模型的技术。通常，教师和学生都是基于Transformer的架构，并利用softmax注意力进行序列建模。然而，在推断时自注意力的二次复杂性仍然是一个显著瓶颈，促使人们探索次二次替代方案，如结构化状态空间模型（SSMs）、线性注意力和递归架构。在本文中，我们系统地评估了从Transformer教师向九种次二次学生架构的知识蒸馏的可转移性。我们的研究旨在确定哪种次二次模型最能与教师学习到的表示相匹配，以及不同的架构约束如何影响蒸馏过程。我们还研究了包括矩阵混合和查询-键-值（QKV）复制在内的智能初始化策略对适应过程的影响。我们在多个自然语言处理基准上的实证结果提供了关于效率与性能之间权衡的见解，并突出了成功将知识转移到次二次架构中的关键因素。 

---
# Accelerating LLM Inference with Flexible N:M Sparsity via A Fully Digital Compute-in-Memory Accelerator 

**Title (ZH)**: 使用全数字计算在内存加速器实现灵活的N:M稀疏性以加快大语言模型推理速度 

**Authors**: Akshat Ramachandran, Souvik Kundu, Arnab Raha, Shamik Kundu, Deepak K. Mathaikutty, Tushar Krishna  

**Link**: [PDF](https://arxiv.org/pdf/2504.14365)  

**Abstract**: Large language model (LLM) pruning with fixed N:M structured sparsity significantly limits the expressivity of the sparse model, yielding sub-optimal performance. In contrast, supporting multiple N:M patterns to provide sparse representational freedom introduces costly overhead in hardware. To address these challenges for LLMs, we first present a flexible layer-wise outlier-density-aware N:M sparsity (FLOW) selection method. FLOW enables the identification of optimal layer-wise N and M values (from a given range) by simultaneously accounting for the presence and distribution of outliers, allowing a higher degree of representational freedom. To deploy sparse models with such N:M flexibility, we then introduce a flexible, low-overhead digital compute-in-memory architecture (FlexCiM). FlexCiM supports diverse sparsity patterns by partitioning a digital CiM (DCiM) macro into smaller sub-macros, which are adaptively aggregated and disaggregated through distribution and merging mechanisms for different N and M values. Extensive experiments on both transformer-based and recurrence-based state space foundation models (SSMs) demonstrate that FLOW outperforms existing alternatives with an accuracy improvement of up to 36%, while FlexCiM achieves up to 1.75x lower inference latency and 1.5x lower energy consumption compared to existing sparse accelerators. Code is available at: this https URL 

**Abstract (ZH)**: 大型语言模型（LLM）中具有固定N:M结构稀疏性的剪枝显著限定了稀疏模型的表达能力，导致性能不佳。相比之下，支持多种N:M模式以提供稀疏表示自由度会引入高昂的硬件开销。为解决这些挑战，我们首先提出了一种灵活的逐层异常密度aware N:M稀疏性（FLOW）选择方法。FLOW能够在同时考虑异常值的存在及其分布的情况下，识别出最优的逐层N和M值（从给定范围内），允许更高的表示自由度。为部署这种N:M灵活性的稀疏模型，我们随后引入了一种灵活的、开销低的数字计算存贮架构（FlexCiM）。FlexCiM通过将一个数字计算存贮宏（DCiM）分割成更小的亚宏，并通过分布和合并机制适应性地聚集和解聚不同的N和M值，支持多种稀疏模式。在基于变换器和基于递归的状态空间基础模型（SSMs）上的 extensive 实验表明，FLOW在现有替代方案中的准确度提高最多可达36%，而FlexCiM相比现有稀疏加速器的推理延迟降低至1.75倍，能耗降低至1.5倍。代码可在以下链接获取：this https URL。 

---
# A Multimodal Recaptioning Framework to Account for Perceptual Diversity in Multilingual Vision-Language Modeling 

**Title (ZH)**: 一种多模态重新描写框架，以考虑多语言视觉-语言模型中的感知多样性 

**Authors**: Kyle Buettner, Jacob Emmerson, Adriana Kovashka  

**Link**: [PDF](https://arxiv.org/pdf/2504.14359)  

**Abstract**: There are many ways to describe, name, and group objects when captioning an image. Differences are evident when speakers come from diverse cultures due to the unique experiences that shape perception. Machine translation of captions has pushed multilingual capabilities in vision-language models (VLMs), but data comes mainly from English speakers, indicating a perceptual bias and lack of model flexibility. In this work, we address this challenge and outline a data-efficient framework to instill multilingual VLMs with greater understanding of perceptual diversity. We specifically propose an LLM-based, multimodal recaptioning strategy that alters the object descriptions of English captions before translation. The greatest benefits are demonstrated in a targeted multimodal mechanism guided by native speaker data. By adding produced rewrites as augmentations in training, we improve on German and Japanese text-image retrieval cases studies (up to +3.5 mean recall overall, +4.7 on non-native error cases). We further propose a mechanism to analyze the specific object description differences across datasets, and we offer insights into cross-dataset and cross-language generalization. 

**Abstract (ZH)**: 描述、命名和分组图像中的对象有多种方式。来自不同文化背景的说话者在表述时因独特的经验而产生感知差异。机器翻译描述提升了视觉-语言模型（VLMs）的多语言能力，但数据主要来自以英语为母语的说话者，这显示出感知偏见和模型灵活性不足的问题。本文旨在应对这一挑战，提出了一种数据高效框架，以增强多语言VLMs对感知多样性理解的能力。我们具体提出了一种基于大规模语言模型的多模态重描述策略，在翻译前修改英语描述。该策略在以母语数据为指导的针对性多模态机制中显示出最大益处。通过将生成的修改作为训练增强，我们在德语和日语文本-图像检索案例研究中取得了改进（总体平均召回率提高3.5%，非母语错误案例提高4.7%）。我们还提出了一种机制来分析不同数据集中的特定对象描述差异，并提供了跨数据集和跨语言泛化的一些见解。 

---
# Integrating LLM-Generated Views into Mean-Variance Optimization Using the Black-Litterman Model 

**Title (ZH)**: 将LLM生成的观点整合到Black-Litterman模型的均值-方差优化中 

**Authors**: Youngbin Lee, Yejin Kim, Suin Kim, Yongjae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.14345)  

**Abstract**: Portfolio optimization faces challenges due to the sensitivity in traditional mean-variance models. The Black-Litterman model mitigates this by integrating investor views, but defining these views remains difficult. This study explores the integration of large language models (LLMs) generated views into portfolio optimization using the Black-Litterman framework. Our method leverages LLMs to estimate expected stock returns from historical prices and company metadata, incorporating uncertainty through the variance in predictions. We conduct a backtest of the LLM-optimized portfolios from June 2024 to February 2025, rebalancing biweekly using the previous two weeks of price data. As baselines, we compare against the S&P 500, an equal-weighted portfolio, and a traditional mean-variance optimized portfolio constructed using the same set of stocks. Empirical results suggest that different LLMs exhibit varying levels of predictive optimism and confidence stability, which impact portfolio performance. The source code and data are available at this https URL. 

**Abstract (ZH)**: 大规模语言模型视角下基于Black-Litterman框架的资产组合优化研究 

---
# Visual Prompting for One-shot Controllable Video Editing without Inversion 

**Title (ZH)**: 无需倒置的一次性可控视频编辑的视觉提示方法 

**Authors**: Zhengbo Zhang, Yuxi Zhou, Duo Peng, Joo-Hwee Lim, Zhigang Tu, De Wen Soh, Lin Geng Foo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14335)  

**Abstract**: One-shot controllable video editing (OCVE) is an important yet challenging task, aiming to propagate user edits that are made -- using any image editing tool -- on the first frame of a video to all subsequent frames, while ensuring content consistency between edited frames and source frames. To achieve this, prior methods employ DDIM inversion to transform source frames into latent noise, which is then fed into a pre-trained diffusion model, conditioned on the user-edited first frame, to generate the edited video. However, the DDIM inversion process accumulates errors, which hinder the latent noise from accurately reconstructing the source frames, ultimately compromising content consistency in the generated edited frames. To overcome it, our method eliminates the need for DDIM inversion by performing OCVE through a novel perspective based on visual prompting. Furthermore, inspired by consistency models that can perform multi-step consistency sampling to generate a sequence of content-consistent images, we propose a content consistency sampling (CCS) to ensure content consistency between the generated edited frames and the source frames. Moreover, we introduce a temporal-content consistency sampling (TCS) based on Stein Variational Gradient Descent to ensure temporal consistency across the edited frames. Extensive experiments validate the effectiveness of our approach. 

**Abstract (ZH)**: 基于视觉提示的一次性可控视频编辑（OCVE）：内容一致性的采样方法 

---
# Expanding the Generative AI Design Space through Structured Prompting and Multimodal Interfaces 

**Title (ZH)**: 通过结构化提示和多模态界面扩展生成式AI设计空间 

**Authors**: Nimisha Karnatak, Adrien Baranes, Rob Marchant, Huinan Zeng, Tríona Butler, Kristen Olson  

**Link**: [PDF](https://arxiv.org/pdf/2504.14320)  

**Abstract**: Text-based prompting remains the dominant interaction paradigm in generative AI, yet it often results in a high-friction experience for novice users, such as small business owners (SBOs), attempting to articulate creative or domain-specific goals for advertising. To investigate this challenge, we conducted a study with six SBOs in the United Kingdom, focusing on their advertising practices and perceptions and usage of AI tools in this context. Our findings surfaced two persistent breakdowns in current generative AI systems: first, the cognitive burden of prompt engineering, as users struggled to translate abstract creative goals into effective textual inputs; and second, the frequent generation of generic outputs that failed to align with users' articulated brand vision. To address these issues, we developed ACAI (AI Co-Creation for Advertising and Inspiration), a multimodal, GenAI-powered advertisement creation tool designed to support novice designers by reimagining the prompt interface. ACAI features a structured, panel-based interface composed of three modules: the Branding Panel, the Audience & Goals Panel, and the Inspiration Board Panel to provide SBOs with outputs that align with their creative vision by reducing prompt ambiguity. This work contributes to HCI research on generative systems by showing how structured interfaces can foreground user-defined context to improve both alignment and promptability in novice workflows. 

**Abstract (ZH)**: 基于文本的提示仍然是生成式AI的主要交互范式，但对于试图为广告 articulated 创造性或领域特定目标的小型企业主（SBOs）来说，往往会带来高摩擦的体验。为了研究这一挑战，我们在英国对六名SBOs进行了研究，关注他们在广告活动中的实践、观念及其对AI工具的使用。我们的研究发现当前生成式AI系统中存在的两大持续性问题：首先，提示工程的认知负担，用户在努力将抽象的创意目标转化为有效的文本输入；其次，生成的输出经常与用户阐明的品牌愿景不一致。为了解决这些问题，我们开发了ACAI（用于广告和灵感的人工智能联合创造），这是一种多模态的、由生成式AI驱动的广告创作工具，旨在通过重塑提示界面来支持初学者设计师。ACAI具有一结构化的面板式界面，由品牌板块、目标与受众板块以及灵感板板块三部分组成，通过减少提示的模糊性，为SBOs提供与其创意愿景相一致的输出。这项工作为生成系统的人机交互研究做出了贡献，展示了结构化界面如何将用户定义的上下文置于首位，以改善初学者工作流中的匹配度和提示能力。 

---
# Learning to Score 

**Title (ZH)**: 学习打分 

**Authors**: Yogev Kriger, Shai Fine  

**Link**: [PDF](https://arxiv.org/pdf/2504.14302)  

**Abstract**: Common machine learning settings range from supervised tasks, where accurately labeled data is accessible, through semi-supervised and weakly-supervised tasks, where target labels are scant or noisy, to unsupervised tasks where labels are unobtainable. In this paper we study a scenario where the target labels are not available but additional related information is at hand. This information, referred to as Side Information, is either correlated with the unknown labels or imposes constraints on the feature space. We formulate the problem as an ensemble of three semantic components: representation learning, side information and metric learning. The proposed scoring model is advantageous for multiple use-cases. For example, in the healthcare domain it can be used to create a severity score for diseases where the symptoms are known but the criteria for the disease progression are not well defined. We demonstrate the utility of the suggested scoring system on well-known benchmark data-sets and bio-medical patient records. 

**Abstract (ZH)**: 常见的机器学习设置包括监督任务、半监督任务、弱监督任务和无监督任务。在监督任务中，有准确标注的数据；半监督和弱监督任务中，目标标签稀少或噪声较大；无监督任务中，标签不可获得。在本文中，我们研究目标标签不可获得但有额外相关信息的情况。这种信息称为辅助信息，它要么与未知标签相关，要么约束特征空间。我们将问题形式化为三个语义组件的组合：表示学习、辅助信息和度量学习。所提出的成绩模型适用于多种应用场景。例如，在医疗领域，它可以用来为已知症状但疾病进展标准不明确的疾病创建严重程度评分。我们通过众所周知的标准数据集和生物医学患者记录展示了所建议评分系统的有效性。 

---
# Balancing Privacy and Action Performance: A Penalty-Driven Approach to Image Anonymization 

**Title (ZH)**: 平衡隐私与行动性能：一种惩罚驱动的图像匿名化方法 

**Authors**: Nazia Aslam, Kamal Nasrollahi  

**Link**: [PDF](https://arxiv.org/pdf/2504.14301)  

**Abstract**: The rapid development of video surveillance systems for object detection, tracking, activity recognition, and anomaly detection has revolutionized our day-to-day lives while setting alarms for privacy concerns. It isn't easy to strike a balance between visual privacy and action recognition performance in most computer vision models. Is it possible to safeguard privacy without sacrificing performance? It poses a formidable challenge, as even minor privacy enhancements can lead to substantial performance degradation. To address this challenge, we propose a privacy-preserving image anonymization technique that optimizes the anonymizer using penalties from the utility branch, ensuring improved action recognition performance while minimally affecting privacy leakage. This approach addresses the trade-off between minimizing privacy leakage and maintaining high action performance. The proposed approach is primarily designed to align with the regulatory standards of the EU AI Act and GDPR, ensuring the protection of personally identifiable information while maintaining action performance. To the best of our knowledge, we are the first to introduce a feature-based penalty scheme that exclusively controls the action features, allowing freedom to anonymize private attributes. Extensive experiments were conducted to validate the effectiveness of the proposed method. The results demonstrate that applying a penalty to anonymizer from utility branch enhances action performance while maintaining nearly consistent privacy leakage across different penalty settings. 

**Abstract (ZH)**: 视频监控系统中基于对象检测、跟踪、行为识别和异常检测的快速发展已经革新了我们的日常生活，同时也引发了隐私担忧。在大多数计算机视觉模型中，要在视觉隐私和行为识别性能之间找到平衡颇不易。是否可以在不牺牲性能的前提下保护隐私？这是一个严峻的挑战，因为即使是微小的隐私增强也可能导致性能大幅度下降。为应对这一挑战，我们提出了一种隐私保护图像匿名化技术，通过使用来自效用分支的惩罚优化匿名器，从而在最小影响隐私泄露的同时提升行为识别性能。该方法旨在平衡减少隐私泄露和保持高水平行为性能之间的权衡。我们主要设计该方法以符合欧盟AI法案和GDPR的监管标准，确保在保护个人可识别信息的同时维持行为性能。据我们所知，我们首次提出了基于特征的惩罚方案，该方案专门控制行为特征，允许对私有属性进行匿名化而不受限制。进行了大量实验证明所提方法的有效性。结果表明，来自效用分支的惩罚应用于匿名器可以提高行为识别性能，同时在不同惩罚设置下保持接近一致的隐私泄露水平。 

---
# Learning and Generating Diverse Residential Load Patterns Using GAN with Weakly-Supervised Training and Weight Selection 

**Title (ZH)**: 使用弱监督训练和权重选择的GAN学习和生成多样化的住宅负荷模式 

**Authors**: Xinyu Liang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14300)  

**Abstract**: The scarcity of high-quality residential load data can pose obstacles for decarbonizing the residential sector as well as effective grid planning and operation. The above challenges have motivated research into generating synthetic load data, but existing methods faced limitations in terms of scalability, diversity, and similarity. This paper proposes a Generative Adversarial Network-based Synthetic Residential Load Pattern (RLP-GAN) generation model, a novel weakly-supervised GAN framework, leveraging an over-complete autoencoder to capture dependencies within complex and diverse load patterns and learn household-level data distribution at scale. We incorporate a model weight selection method to address the mode collapse problem and generate load patterns with high diversity. We develop a holistic evaluation method to validate the effectiveness of RLP-GAN using real-world data of 417 households. The results demonstrate that RLP-GAN outperforms state-of-the-art models in capturing temporal dependencies and generating load patterns with higher similarity to real data. Furthermore, we have publicly released the RLP-GAN generated synthetic dataset, which comprises one million synthetic residential load pattern profiles. 

**Abstract (ZH)**: 基于生成对抗网络的合成住宅负荷模式生成模型（RLP-GAN）：一种新颖的弱监督GAN框架 

---
# Experience-based Refinement of Task Planning Knowledge in Autonomous Robots 

**Title (ZH)**: 基于经验的任务规划知识精炼在自主机器人中的应用 

**Authors**: Hadeel Jazzaa, Thomas McCluskey, David Peebles  

**Link**: [PDF](https://arxiv.org/pdf/2504.14259)  

**Abstract**: The requirement for autonomous robots to exhibit higher-level cognitive skills by planning and adapting in an ever-changing environment is indeed a great challenge for the AI community. Progress has been made in the automated planning community on refinement and repair of an agent's symbolic knowledge to do task planning in an incomplete or changing environmental model, but these advances up to now have not been transferred to real physical robots. This paper demonstrates how a physical robot can be capable of adapting its symbolic knowledge of the environment, by using experiences in robot action execution to drive knowledge refinement and hence to improve the success rate of the task plans the robot creates. To implement more robust planning systems, we propose a method for refining domain knowledge to improve the knowledge on which intelligent robot behavior is based. This architecture has been implemented and evaluated using a NAO robot. The refined knowledge leads to the future synthesis of task plans which demonstrate decreasing rates of failure over time as faulty knowledge is removed or adjusted. 

**Abstract (ZH)**: 自主机器人在不断变化环境中表现出更高层次认知能力的要求确实是AI社区的一大挑战。尽管自动规划领域在细化和修复代理符号知识以进行任务规划方面取得了进展，但在不完整或变化的环境模型中，这些进展尚未转移到真实物理机器人上。本文展示了如何通过利用机器人动作执行的经验来驱动知识细化，从而使物理机器人能够适应其对环境的符号知识，进而提高机器人创建任务计划的成功率。为了构建更稳健的规划系统，我们提出了一种方法，用于细化领域知识，以提高基于智能机器人行为的知识基础。该架构已在NAO机器人上实现并进行了评估。细化的知识导致未来合成的任务计划随着时间的推移失效率降低，无效知识被删除或调整。 

---
# SimplifyMyText: An LLM-Based System for Inclusive Plain Language Text Simplification 

**Title (ZH)**: SimplifyMyText：基于LLM的包容性简化文本系统 

**Authors**: Michael Färber, Parisa Aghdam, Kyuri Im, Mario Tawfelis, Hardik Ghoshal  

**Link**: [PDF](https://arxiv.org/pdf/2504.14223)  

**Abstract**: Text simplification is essential for making complex content accessible to diverse audiences who face comprehension challenges. Yet, the limited availability of simplified materials creates significant barriers to personal and professional growth and hinders social inclusion. Although researchers have explored various methods for automatic text simplification, none fully leverage large language models (LLMs) to offer tailored customization for different target groups and varying levels of simplicity. Moreover, despite its proven benefits for both consumers and organizations, the well-established practice of plain language remains underutilized. In this paper, we this https URL, the first system designed to produce plain language content from multiple input formats, including typed text and file uploads, with flexible customization options for diverse audiences. We employ GPT-4 and Llama-3 and evaluate outputs across multiple metrics. Overall, our work contributes to research on automatic text simplification and highlights the importance of tailored communication in promoting inclusivity. 

**Abstract (ZH)**: 文本简化对于使复杂内容对面临理解挑战的多样化受众群体变得易于访问是必不可少的。然而，简化材料的有限可用性为个人和职业成长造成了重大障碍，并妨碍了社会包容性。尽管研究人员探索了各种自动文本简化方法，但现有的方法未能充分利用大型语言模型（LLMs）来为不同目标群体和不同的简化程度提供量身定制的服务。此外，尽管简洁语言对消费者和组织都有明显的好处，但这一成熟的做法仍被严重低估。在本文中，我们介绍了this https URL，这是第一个能够从多种输入格式（包括键盘输入和文件上传）生成简洁语言内容的系统，并提供多种定制选项以适应不同的受众群体。我们使用GPT-4和Llama-3进行评估，并通过多个指标来评估输出结果。总体而言，我们的工作为自动文本简化研究做出了贡献，并突显了量身定制沟通在促进包容性中的重要性。 

---
# Decomposition-based multi-scale transformer framework for time series anomaly detection 

**Title (ZH)**: 基于分解的多尺度变换器框架的时间序列异常检测 

**Authors**: Wenxin Zhang, Cuicui Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14206)  

**Abstract**: Time series anomaly detection is crucial for maintaining stable systems. Existing methods face two main challenges. First, it is difficult to directly model the dependencies of diverse and complex patterns within the sequences. Second, many methods that optimize parameters using mean squared error struggle with noise in the time series, leading to performance deterioration. To address these challenges, we propose a transformer-based framework built on decomposition (TransDe) for multivariate time series anomaly detection. The key idea is to combine the strengths of time series decomposition and transformers to effectively learn the complex patterns in normal time series data. A multi-scale patch-based transformer architecture is proposed to exploit the representative dependencies of each decomposed component of the time series. Furthermore, a contrastive learn paradigm based on patch operation is proposed, which leverages KL divergence to align the positive pairs, namely the pure representations of normal patterns between different patch-level views. A novel asynchronous loss function with a stop-gradient strategy is further introduced to enhance the performance of TransDe effectively. It can avoid time-consuming and labor-intensive computation costs in the optimization process. Extensive experiments on five public datasets are conducted and TransDe shows superiority compared with twelve baselines in terms of F1 score. Our code is available at this https URL. 

**Abstract (ZH)**: 基于分解的Transformer框架在多变元时间序列异常检测中的应用 

---
# Dual-channel Heterophilic Message Passing for Graph Fraud Detection 

**Title (ZH)**: 双重通道异类消息传递用于图欺诈检测 

**Authors**: Wenxin Zhang, Jingxing Zhong, Guangzhen Yao, Renda Han, Xiaojian Lin, Zeyu Zhang, Cuicui Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14205)  

**Abstract**: Fraudulent activities have significantly increased across various domains, such as e-commerce, online review platforms, and social networks, making fraud detection a critical task. Spatial Graph Neural Networks (GNNs) have been successfully applied to fraud detection tasks due to their strong inductive learning capabilities. However, existing spatial GNN-based methods often enhance the graph structure by excluding heterophilic neighbors during message passing to align with the homophilic bias of GNNs. Unfortunately, this approach can disrupt the original graph topology and increase uncertainty in predictions. To address these limitations, this paper proposes a novel framework, Dual-channel Heterophilic Message Passing (DHMP), for fraud detection. DHMP leverages a heterophily separation module to divide the graph into homophilic and heterophilic subgraphs, mitigating the low-pass inductive bias of traditional GNNs. It then applies shared weights to capture signals at different frequencies independently and incorporates a customized sampling strategy for training. This allows nodes to adaptively balance the contributions of various signals based on their labels. Extensive experiments on three real-world datasets demonstrate that DHMP outperforms existing methods, highlighting the importance of separating signals with different frequencies for improved fraud detection. The code is available at this https URL. 

**Abstract (ZH)**: 欺诈活动在电子商务、在线评价平台和社会网络等领域显著增加，使欺诈检测成为一项关键任务。空间图神经网络（GNNs）由于其强大的归纳学习能力，在欺诈检测任务中取得了成功应用。然而，现有的基于空间GNN的方法常常在消息传递过程中通过排除异ophilic邻居来增强图结构，以符合GNN的同ophilic偏见。不幸的是，这种方法可能会破坏原始图拓扑结构，增加预测的不确定性。为了解决这些局限性，本文提出了一种新的框架，双通道异ophilic消息传递（DHMP）方法，用于欺诈检测。DHMP利用一个异ophilic分离模块将图划分为同ophilic和异ophilic子图，以减轻传统GNN的低通归纳偏差。然后，它使用共享权重独立捕获不同频率的信号，并结合一种定制的采样策略进行训练。这使节点能够根据其标签适应性地平衡各种信号的贡献。在三个真实世界数据集上的广泛实验表明，DHMP优于现有方法，突出了分离不同频率信号以提高欺诈检测效果的重要性。代码可在此处获取。 

---
# DConAD: A Differencing-based Contrastive Representation Learning Framework for Time Series Anomaly Detection 

**Title (ZH)**: DConAD：一种基于差异对比的时序异常检测表示学习框架 

**Authors**: Wenxin Zhang, Xiaojian Lin, Wenjun Yu, Guangzhen Yao, jingxiang Zhong, Yu Li, Renda Han, Songcheng Xu, Hao Shi, Cuicui Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.14204)  

**Abstract**: Time series anomaly detection holds notable importance for risk identification and fault detection across diverse application domains. Unsupervised learning methods have become popular because they have no requirement for labels. However, due to the challenges posed by the multiplicity of abnormal patterns, the sparsity of anomalies, and the growth of data scale and complexity, these methods often fail to capture robust and representative dependencies within the time series for identifying anomalies. To enhance the ability of models to capture normal patterns of time series and avoid the retrogression of modeling ability triggered by the dependencies on high-quality prior knowledge, we propose a differencing-based contrastive representation learning framework for time series anomaly detection (DConAD). Specifically, DConAD generates differential data to provide additional information about time series and utilizes transformer-based architecture to capture spatiotemporal dependencies, which enhances the robustness of unbiased representation learning ability. Furthermore, DConAD implements a novel KL divergence-based contrastive learning paradigm that only uses positive samples to avoid deviation from reconstruction and deploys the stop-gradient strategy to compel convergence. Extensive experiments on five public datasets show the superiority and effectiveness of DConAD compared with nine baselines. The code is available at this https URL. 

**Abstract (ZH)**: 基于差分的对比表示学习框架：时间序列异常检测（DConAD） 

---
# Learning Joint ID-Textual Representation for ID-Preserving Image Synthesis 

**Title (ZH)**: ID保真的联合身份-文本表示学习 

**Authors**: Zichuan Liu, Liming Jiang, Qing Yan, Yumin Jia, Hao Kang, Xin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14202)  

**Abstract**: We propose a novel framework for ID-preserving generation using a multi-modal encoding strategy rather than injecting identity features via adapters into pre-trained models. Our method treats identity and text as a unified conditioning input. To achieve this, we introduce FaceCLIP, a multi-modal encoder that learns a joint embedding space for both identity and textual semantics. Given a reference face and a text prompt, FaceCLIP produces a unified representation that encodes both identity and text, which conditions a base diffusion model to generate images that are identity-consistent and text-aligned. We also present a multi-modal alignment algorithm to train FaceCLIP, using a loss that aligns its joint representation with face, text, and image embedding spaces. We then build FaceCLIP-SDXL, an ID-preserving image synthesis pipeline by integrating FaceCLIP with Stable Diffusion XL (SDXL). Compared to prior methods, FaceCLIP-SDXL enables photorealistic portrait generation with better identity preservation and textual relevance. Extensive experiments demonstrate its quantitative and qualitative superiority. 

**Abstract (ZH)**: 我们提出了一种新的框架，采用多模态编码策略实现身份保留生成，而不是通过适配器将身份特征注入预训练模型。我们的方法将身份和文本视为统一的条件输入。为此，我们引入了FaceCLIP，这是一种多模态编码器，学习身份和文本语义的联合嵌入空间。给定一个参考人脸和文本提示，FaceCLIP生成一个统一表示，同时编码身份和文本，用于条件基扩散模型生成与身份一致且与文本对齐的图像。我们还提出了一种多模态对齐算法来训练FaceCLIP，使用一种损失函数，使其联合表示与人脸、文本和图像嵌入空间对齐。然后，我们构建了FaceCLIP-SDXL，这是一种通过将FaceCLIP与Stable Diffusion XL (SDXL) 结合的具有身份保留的图像合成管线。与先前的方法相比，FaceCLIP-SDXL 能够生成更具真实感的肖像，同时保持更好的身份一致性与文本相关性。大量实验证明了其在定量和定性上的优越性。 

---
# Enhancing Multimodal In-Context Learning for Image Classification through Coreset Optimization 

**Title (ZH)**: 通过核心样本优化提升多模态上下文学习的图像分类性能 

**Authors**: Huiyi Chen, Jiawei Peng, Kaihua Tang, Xin Geng, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14200)  

**Abstract**: In-context learning (ICL) enables Large Vision-Language Models (LVLMs) to adapt to new tasks without parameter updates, using a few demonstrations from a large support set. However, selecting informative demonstrations leads to high computational and memory costs. While some methods explore selecting a small and representative coreset in the text classification, evaluating all support set samples remains costly, and discarded samples lead to unnecessary information loss. These methods may also be less effective for image classification due to differences in feature spaces. Given these limitations, we propose Key-based Coreset Optimization (KeCO), a novel framework that leverages untapped data to construct a compact and informative coreset. We introduce visual features as keys within the coreset, which serve as the anchor for identifying samples to be updated through different selection strategies. By leveraging untapped samples from the support set, we update the keys of selected coreset samples, enabling the randomly initialized coreset to evolve into a more informative coreset under low computational cost. Through extensive experiments on coarse-grained and fine-grained image classification benchmarks, we demonstrate that KeCO effectively enhances ICL performance for image classification task, achieving an average improvement of more than 20\%. Notably, we evaluate KeCO under a simulated online scenario, and the strong performance in this scenario highlights the practical value of our framework for resource-constrained real-world scenarios. 

**Abstract (ZH)**: 基于键的核心集优化（KeCO）：提升图像分类的上下文学习性能 

---
# A Physics-guided Multimodal Transformer Path to Weather and Climate Sciences 

**Title (ZH)**: 物理指导的多模态变压器路径在天气与气候科学中的应用 

**Authors**: Jing Han, Hanting Chen, Kai Han, Xiaomeng Huang, Yongyun Hu, Wenjun Xu, Dacheng Tao, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14174)  

**Abstract**: With the rapid development of machine learning in recent years, many problems in meteorology can now be addressed using AI models. In particular, data-driven algorithms have significantly improved accuracy compared to traditional methods. Meteorological data is often transformed into 2D images or 3D videos, which are then fed into AI models for learning. Additionally, these models often incorporate physical signals, such as temperature, pressure, and wind speed, to further enhance accuracy and interpretability. In this paper, we review several representative AI + Weather/Climate algorithms and propose a new paradigm where observational data from different perspectives, each with distinct physical meanings, are treated as multimodal data and integrated via transformers. Furthermore, key weather and climate knowledge can be incorporated through regularization techniques to further strengthen the model's capabilities. This new paradigm is versatile and can address a variety of tasks, offering strong generalizability. We also discuss future directions for improving model accuracy and interpretability. 

**Abstract (ZH)**: 近年来，随着机器学习的快速发展，许多气象问题现在可以使用AI模型来解决。特别是数据驱动的算法相比传统方法显著提高了准确性。气象数据通常被转换为2D图像或3D视频，然后输入到AI模型中进行学习。此外，这些模型还常常结合物理信号，如温度、压力和风速，以进一步提高准确性和可解释性。在本文中，我们回顾了几种代表性的AI + 气象/气候算法，并提出了一种新范式，即将来自不同视角的观测数据，每种数据具有不同的物理意义，视为多模态数据并通过变换器进行整合。此外，通过正则化技术可以嵌入关键的气象和气候知识，以进一步增强模型的能力。该新范式具有广泛的适用性，可以解决多种任务，提供强大的泛化能力。我们还讨论了改进模型准确性和可解释性的未来方向。 

---
# Breaking the Diffraction Barrier for Passive Sources: Parameter-Decoupled Superresolution Assisted by Physics-Informed Machine Learning 

**Title (ZH)**: 突破衍射极限的被动源：参数解耦超分辨辅助下的物理知情机器学习 

**Authors**: Abdelali Sajia, Bilal Benzimoun, Pawan Khatiwada, Guogan Zhao, Xiao-Feng Qian  

**Link**: [PDF](https://arxiv.org/pdf/2504.14156)  

**Abstract**: We present a parameter-decoupled superresolution framework for estimating sub-wavelength separations of passive two-point sources without requiring prior knowledge or control of the source. Our theoretical foundation circumvents the need to estimate multiple challenging parameters such as partial coherence, brightness imbalance, random relative phase, and photon statistics. A physics-informed machine learning (ML) model (trained with a standard desktop workstation), synergistically integrating this theory, further addresses practical imperfections including background noise, photon loss, and centroid/orientation misalignment. The integrated parameter-decoupling superresolution method achieves resolution 14 and more times below the diffraction limit (corresponding to ~ 13.5 nm in optical microscopy) on experimentally generated realistic images with >82% fidelity, performance rivaling state-of-the-art techniques for actively controllable sources. Critically, our method's robustness against source parameter variability and source-independent noises enables potential applications in realistic scenarios where source control is infeasible, such as astrophysical imaging, live-cell microscopy, and quantum metrology. This work bridges a critical gap between theoretical superresolution limits and practical implementations for passive systems. 

**Abstract (ZH)**: 无源两点源超分辨框架：无需先验知识或源控制的亚波长分离估计 

---
# SConU: Selective Conformal Uncertainty in Large Language Models 

**Title (ZH)**: 选择性齐性不确定性在大型语言模型中 

**Authors**: Zhiyuan Wang, Qingni Wang, Yue Zhang, Tianlong Chen, Xiaofeng Zhu, Xiaoshuang Shi, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.14154)  

**Abstract**: As large language models are increasingly utilized in real-world applications, guarantees of task-specific metrics are essential for their reliable deployment. Previous studies have introduced various criteria of conformal uncertainty grounded in split conformal prediction, which offer user-specified correctness coverage. However, existing frameworks often fail to identify uncertainty data outliers that violate the exchangeability assumption, leading to unbounded miscoverage rates and unactionable prediction sets. In this paper, we propose a novel approach termed Selective Conformal Uncertainty (SConU), which, for the first time, implements significance tests, by developing two conformal p-values that are instrumental in determining whether a given sample deviates from the uncertainty distribution of the calibration set at a specific manageable risk level. Our approach not only facilitates rigorous management of miscoverage rates across both single-domain and interdisciplinary contexts, but also enhances the efficiency of predictions. Furthermore, we comprehensively analyze the components of the conformal procedures, aiming to approximate conditional coverage, particularly in high-stakes question-answering tasks. 

**Abstract (ZH)**: 随着大型语言模型在实际应用中的日益普及，任务特定指标的保证对于其可靠部署至关重要。先前的研究引入了基于分割一致性预测的各种符合性不确定性准则，这些准则提供了用户指定的正确性覆盖范围。然而，现有的框架往往无法识别违反可交换性假设的不确定性数据离群值，导致未界定的覆盖误差率和不可行动的预测集。在本文中，我们提出了一种新颖的方法，即选择性一致性不确定性（SConU），这是首次通过开发两种在特定可管理风险水平上确定给定样本是否偏离校准集合的不确定性分布的重要一致性p值来实施显著性检验。我们的方法不仅在单域和跨学科背景下促进了严格的覆盖误差率管理，还提高了预测效率。此外，我们全面分析了一致性方法的各个组成部分，旨在近似条件覆盖，特别是在高风险问答任务中。 

---
# Locate 3D: Real-World Object Localization via Self-Supervised Learning in 3D 

**Title (ZH)**: Locate 3D：通过3D自主学习进行真实世界物体定位 

**Authors**: Sergio Arnaud, Paul McVay, Ada Martin, Arjun Majumdar, Krishna Murthy Jatavallabhula, Phillip Thomas, Ruslan Partsey, Daniel Dugas, Abha Gejji, Alexander Sax, Vincent-Pierre Berges, Mikael Henaff, Ayush Jain, Ang Cao, Ishita Prasad, Mrinal Kalakrishnan, Michael Rabbat, Nicolas Ballas, Mido Assran, Oleksandr Maksymets, Aravind Rajeswaran, Franziska Meier  

**Link**: [PDF](https://arxiv.org/pdf/2504.14151)  

**Abstract**: We present LOCATE 3D, a model for localizing objects in 3D scenes from referring expressions like "the small coffee table between the sofa and the lamp." LOCATE 3D sets a new state-of-the-art on standard referential grounding benchmarks and showcases robust generalization capabilities. Notably, LOCATE 3D operates directly on sensor observation streams (posed RGB-D frames), enabling real-world deployment on robots and AR devices. Key to our approach is 3D-JEPA, a novel self-supervised learning (SSL) algorithm applicable to sensor point clouds. It takes as input a 3D pointcloud featurized using 2D foundation models (CLIP, DINO). Subsequently, masked prediction in latent space is employed as a pretext task to aid the self-supervised learning of contextualized pointcloud features. Once trained, the 3D-JEPA encoder is finetuned alongside a language-conditioned decoder to jointly predict 3D masks and bounding boxes. Additionally, we introduce LOCATE 3D DATASET, a new dataset for 3D referential grounding, spanning multiple capture setups with over 130K annotations. This enables a systematic study of generalization capabilities as well as a stronger model. 

**Abstract (ZH)**: LOCATE 3D：一种基于引用表达在3D场景中定位对象的模型 

---
# Walk the Talk? Measuring the Faithfulness of Large Language Model Explanations 

**Title (ZH)**: 说走就走？大规模语言模型解释的忠实性度量 

**Authors**: Katie Matton, Robert Osazuwa Ness, John Guttag, Emre Kıcıman  

**Link**: [PDF](https://arxiv.org/pdf/2504.14150)  

**Abstract**: Large language models (LLMs) are capable of generating plausible explanations of how they arrived at an answer to a question. However, these explanations can misrepresent the model's "reasoning" process, i.e., they can be unfaithful. This, in turn, can lead to over-trust and misuse. We introduce a new approach for measuring the faithfulness of LLM explanations. First, we provide a rigorous definition of faithfulness. Since LLM explanations mimic human explanations, they often reference high-level concepts in the input question that purportedly influenced the model. We define faithfulness in terms of the difference between the set of concepts that LLM explanations imply are influential and the set that truly are. Second, we present a novel method for estimating faithfulness that is based on: (1) using an auxiliary LLM to modify the values of concepts within model inputs to create realistic counterfactuals, and (2) using a Bayesian hierarchical model to quantify the causal effects of concepts at both the example- and dataset-level. Our experiments show that our method can be used to quantify and discover interpretable patterns of unfaithfulness. On a social bias task, we uncover cases where LLM explanations hide the influence of social bias. On a medical question answering task, we uncover cases where LLM explanations provide misleading claims about which pieces of evidence influenced the model's decisions. 

**Abstract (ZH)**: 大型语言模型解释的忠实度测量：一种新方法及其应用 

---
# HF4Rec: Human-Like Feedback-Driven Optimization Framework for Explainable Recommendation 

**Title (ZH)**: HF4Rec：基于人类反馈驱动优化的可解释推荐框架 

**Authors**: Jiakai Tang, Jingsen Zhang, Zihang Tian, Xueyang Feng, Lei Wang, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.14147)  

**Abstract**: Recent advancements in explainable recommendation have greatly bolstered user experience by elucidating the decision-making rationale. However, the existing methods actually fail to provide effective feedback signals for potentially better or worse generated explanations due to their reliance on traditional supervised learning paradigms in sparse interaction data. To address these issues, we propose a novel human-like feedback-driven optimization framework. This framework employs a dynamic interactive optimization mechanism for achieving human-centered explainable requirements without incurring high labor costs. Specifically, we propose to utilize large language models (LLMs) as human simulators to predict human-like feedback for guiding the learning process. To enable the LLMs to deeply understand the task essence and meet user's diverse personalized requirements, we introduce a human-induced customized reward scoring method, which helps stimulate the language understanding and logical reasoning capabilities of LLMs. Furthermore, considering the potential conflicts between different perspectives of explanation quality, we introduce a principled Pareto optimization that transforms the multi-perspective quality enhancement task into a multi-objective optimization problem for improving explanation performance. At last, to achieve efficient model training, we design an off-policy optimization pipeline. By incorporating a replay buffer and addressing the data distribution biases, we can effectively improve data utilization and enhance model generality. Extensive experiments on four datasets demonstrate the superiority of our approach. 

**Abstract (ZH)**: 近年来可解释推荐系统的进步显著提升了用户体验，通过阐明决策原理。然而，现有方法实际上由于依赖传统的监督学习范式，在稀疏交互数据中无法有效提供对生成解释更好或更差的反馈信号。为解决这些问题，我们提出了一种新颖的人工智能反馈驱动优化框架。该框架采用动态交互优化机制，以实现以用户为中心的可解释需求，而无需高人力成本。具体而言，我们建议使用大型语言模型（LLMs）作为人类模拟器，预测人类-like反馈以指导学习过程。为了使LLMs深刻理解任务本质并满足用户多样化的个性化需求，我们引入了一种由人类引导的定制奖励评分方法，这有助于激发LLMs的语言理解和逻辑推理能力。此外，考虑到不同解释质量视角之间的潜在冲突，我们引入了一种原则性的帕累托优化方法，将多视角质量增强任务转化为多目标优化问题，以提高解释性能。最后，为实现高效的模型训练，我们设计了一种离策优化管道。通过引入重播缓冲区并解决数据分布偏见，我们可以有效提高数据利用率并增强模型的一般性。在四个数据集上的广泛实验表明了我们方法的优越性。 

---
# PipeWeaver: Addressing Data Dynamicity in Large Multimodal Model Training with Dynamic Interleaved Pipeline 

**Title (ZH)**: PipeWeaver: 通过动态插值管道解决大型多模态模型训练中的数据动态性问题 

**Authors**: Zhenliang Xue, Hanpeng Hu, Xing Chen, Yimin Jiang, Yixin Song, Zeyu Mi, Yibo Zhu, Daxin Jiang, Yubin Xia, Haibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.14145)  

**Abstract**: Large multimodal models (LMMs) have demonstrated excellent capabilities in both understanding and generation tasks with various modalities. While these models can accept flexible combinations of input data, their training efficiency suffers from two major issues: pipeline stage imbalance caused by heterogeneous model architectures, and training data dynamicity stemming from the diversity of multimodal data.
In this paper, we present PipeWeaver, a dynamic pipeline scheduling framework designed for LMM training. The core of PipeWeaver is dynamic interleaved pipeline, which searches for pipeline schedules dynamically tailored to current training batches. PipeWeaver addresses issues of LMM training with two techniques: adaptive modality-aware partitioning and efficient pipeline schedule search within a hierarchical schedule space. Meanwhile, PipeWeaver utilizes SEMU (Step Emulator), a training simulator for multimodal models, for accurate performance estimations, accelerated by spatial-temporal subgraph reuse to improve search efficiency. Experiments show that PipeWeaver can enhance LMM training efficiency by up to 97.3% compared to state-of-the-art systems, and demonstrate excellent adaptivity to LMM training's data dynamicity. 

**Abstract (ZH)**: 大型多模态模型（LMMs）在理解和生成任务中表现出卓越的能力，能够处理多种模态的数据。然而，这些模型的训练效率因异构模型架构导致的管道阶段不平衡以及多模态数据多样性带来的训练数据动态性而受到影响。

本文提出了一种名为PipeWeaver的动态管道调度框架，旨在优化LMM的训练。PipeWeaver的核心是动态交织的管道，它会根据不同训练批次自动生成最合适的管道调度策略。PipeWeaver通过两种技术解决了LMM训练中的问题：自适应模态感知分割和层次化调度空间内的高效管道调度搜索。同时，PipeWeaver利用了基于步骤模拟器(SEMU)的训练模拟器进行准确的性能估计，并通过时空子图重用提高搜索效率。实验结果显示，与当前最先进的系统相比，PipeWeaver可以提高LMM训练效率高达97.3%，并且能够很好地适应LMM训练中的数据动态性。 

---
# ThyroidEffi 1.0: A Cost-Effective System for High-Performance Multi-Class Thyroid Carcinoma Classification 

**Title (ZH)**: ThyroidEffi 1.0: 一种高性能多类甲状腺癌分类的成本-effective系统 

**Authors**: Hai Pham-Ngoc, De Nguyen-Van, Dung Vu-Tien, Phuong Le-Hong  

**Link**: [PDF](https://arxiv.org/pdf/2504.14139)  

**Abstract**: Background: Automated classification of thyroid fine needle aspiration biopsy (FNAB) images faces challenges in limited data, inter-observer variability, and computational cost. Efficient, interpretable models are crucial for clinical support. Objective: To develop and externally validate a deep learning system for the multi-class classification of thyroid FNAB images into three key categories that directly guide post-biopsy treatment decisions in Vietnam: benign (B2), suspicious for malignancy (B5), and malignant (B6), while achieving high diagnostic accuracy with low computational overhead. Methods: Our framework features: (1) YOLOv10-based cell cluster detection for informative sub-region extraction and noise reduction; (2) a curriculum learning-inspired protocol sequencing localized crops to full images for multi-scale feature capture; (3) adaptive lightweight EfficientNetB0 (4 millions parameters) selection balancing performance and efficiency; and (4) a Transformer-inspired module for multi-scale, multi-region analysis. External validation used 1,015 independent FNAB images. Results: ThyroidEffi Basic achieved a macro F1 of 89.19\% and AUCs of 0.98 (B2), 0.95 (B5), and 0.96 (B6) on the internal test set. External validation yielded AUCs of 0.9495 (B2), 0.7436 (B5), and 0.8396 (B6). ThyroidEffi Premium improved macro F1 to 89.77\%. Grad-CAM highlighted key diagnostic regions, confirming interpretability. The system processed 1000 cases in 30 seconds, demonstrating feasibility on widely accessible hardware like a 12-core CPU. Conclusions: This work demonstrates that high-accuracy, interpretable thyroid FNAB image classification is achievable with minimal computational demands. 

**Abstract (ZH)**: 背景:甲状腺细针穿刺活检（FNAB）图像的自动分类面临数据有限、观察者间变异性以及计算成本高的挑战。高效的可解释模型对于临床支持至关重要。目的:开发并外部验证一个深度学习系统，用于将甲状腺FNAB图像分为直接指导术后治疗决策的三个关键类别：良性（B2）、可疑恶性（B5）和恶性（B6），同时实现高诊断准确性并具备低计算开销。方法:我们的框架包括：（1）基于YOLOv10的细胞簇检测，用于信息子区域提取和噪声减少；（2）基于曲率学习的协议，按局部裁剪到全图像的序列进行多尺度特征捕捉；（3）自适应的轻量级EfficientNetB0（400万个参数）选择，平衡性能和效率；（4）基于Transformer的设计模块进行多尺度、多区域分析。外部验证使用了1015张独立的FNAB图像。结果: ThyroidEffi Basic在内部测试集上实现了宏F1值89.19%和AUC值分别为0.98（B2）、0.95（B5）和0.96（B6）。外部验证AUC值分别为0.9495（B2）、0.7436（B5）和0.8396（B6）。ThyroidEffi Premium将宏F1提高到了89.77%。Grad-CAM高亮了关键诊断区域，证实了系统的可解释性。该系统在12核CPU等广泛可用的硬件上每秒处理1000个案例，展示了其实现的可行性。结论:本研究证明，即使在低计算需求下，高准确度和可解释的甲状腺FNAB图像分类也是可行的。 

---
# Personalized News Recommendation with Multi-granularity Candidate-aware User Modeling 

**Title (ZH)**: 多粒度候选意识用户建模的个性化新闻推荐 

**Authors**: Qiang Li, Xinze Lin, Shenghao Lv, Faliang Huang, Xiangju Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.14130)  

**Abstract**: Matching candidate news with user interests is crucial for personalized news recommendations. Most existing methods can represent a user's reading interests through a single profile based on clicked news, which may not fully capture the diversity of user interests. Although some approaches incorporate candidate news or topic information, they remain insufficient because they neglect the multi-granularity relatedness between candidate news and user interests. To address this, this study proposed a multi-granularity candidate-aware user modeling framework that integrated user interest features across various levels of granularity. It consisted of two main components: candidate news encoding and user modeling. A news textual information extractor and a knowledge-enhanced entity information extractor can capture candidate news features, and word-level, entity-level, and news-level candidate-aware mechanisms can provide a comprehensive representation of user interests. Extensive experiments on a real-world dataset demonstrated that the proposed model could significantly outperform baseline models. 

**Abstract (ZH)**: 基于多粒度候选新闻的用户建模框架：捕获用户兴趣的多样性 

---
# Exploring Language Patterns of Prompts in Text-to-Image Generation and Their Impact on Visual Diversity 

**Title (ZH)**: 探索文本生成图像中提示的语言模式及其对视觉多样性的影响 

**Authors**: Maria-Teresa De Rosa Palmini, Eva Cetinic  

**Link**: [PDF](https://arxiv.org/pdf/2504.14125)  

**Abstract**: Following the initial excitement, Text-to-Image (TTI) models are now being examined more critically. While much of the discourse has focused on biases and stereotypes embedded in large-scale training datasets, the sociotechnical dynamics of user interactions with these models remain underexplored. This study examines the linguistic and semantic choices users make when crafting prompts and how these choices influence the diversity of generated outputs. Analyzing over six million prompts from the Civiverse dataset on the CivitAI platform across seven months, we categorize users into three groups based on their levels of linguistic experimentation: consistent repeaters, occasional repeaters, and non-repeaters. Our findings reveal that as user participation grows over time, prompt language becomes increasingly homogenized through the adoption of popular community tags and descriptors, with repeated prompts comprising 40-50% of submissions. At the same time, semantic similarity and topic preferences remain relatively stable, emphasizing common subjects and surface aesthetics. Using Vendi scores to quantify visual diversity, we demonstrate a clear correlation between lexical similarity in prompts and the visual similarity of generated images, showing that linguistic repetition reinforces less diverse representations. These findings highlight the significant role of user-driven factors in shaping AI-generated imagery, beyond inherent model biases, and underscore the need for tools and practices that encourage greater linguistic and thematic experimentation within TTI systems to foster more inclusive and diverse AI-generated content. 

**Abstract (ZH)**: Text-to-Image模型：用户互动中的语言与语义选择及其对生成输出多样性的影响 

---
# Longitudinal Study on Social and Emotional Use of AI Conversational Agent 

**Title (ZH)**: longitudinal研究：社会情感使用中的AI对话代理 

**Authors**: Mohit Chandra, Javier Hernandez, Gonzalo Ramos, Mahsa Ershadi, Ananya Bhattacharjee, Judith Amores, Ebele Okoli, Ann Paradiso, Shahed Warreth, Jina Suh  

**Link**: [PDF](https://arxiv.org/pdf/2504.14112)  

**Abstract**: Development in digital technologies has continuously reshaped how individuals seek and receive social and emotional support. While online platforms and communities have long served this need, the increased integration of general-purpose conversational AI into daily lives has introduced new dynamics in how support is provided and experienced. Existing research has highlighted both benefits (e.g., wider access to well-being resources) and potential risks (e.g., over-reliance) of using AI for support seeking. In this five-week, exploratory study, we recruited 149 participants divided into two usage groups: a baseline usage group (BU, n=60) that used the internet and AI as usual, and an active usage group (AU, n=89) encouraged to use one of four commercially available AI tools (Microsoft Copilot, Google Gemini, PI AI, ChatGPT) for social and emotional interactions. Our analysis revealed significant increases in perceived attachment towards AI (32.99 percentage points), perceived AI empathy (25.8 p.p.), and motivation to use AI for entertainment (22.90 p.p.) among the AU group. We also observed that individual differences (e.g., gender identity, prior AI usage) influenced perceptions of AI empathy and attachment. Lastly, the AU group expressed higher comfort in seeking personal help, managing stress, obtaining social support, and talking about health with AI, indicating potential for broader emotional support while highlighting the need for safeguards against problematic usage. Overall, our exploratory findings underscore the importance of developing consumer-facing AI tools that support emotional well-being responsibly, while empowering users to understand the limitations of these tools. 

**Abstract (ZH)**: 数字技术的发展不断重塑个体寻求和接收社会和情感支持的方式。在线平台和社区长期服务于这一需求，而日常生活中的通用对话型AI集成增加，引入了支持提供和体验的新动态。现有研究强调了使用AI寻求支持的益处（如更广泛的福祉资源获取）和潜在风险（如过度依赖）。在为期五周的探索性研究中，我们招募了149名参与者，分为两组使用群体：常规使用组（BU，n=60）继续正常使用互联网和AI，活跃使用组（AU，n=89）被鼓励使用四种商用可用的AI工具（Microsoft Copilot、Google Gemini、PI AI、ChatGPT）进行社交和情感互动。我们的分析显示，活跃使用组（AU组）对AI的情感依附感知提高了32.99个百分点，感知到的AI同理心提高25.8个百分点，以及使用AI进行娱乐的动力提高22.90个百分点。我们还发现，个体差异（如性别认同、先前的AI使用情况）影响了对AI同理心和情感依附的感知。最后，活跃使用组表示在寻求个人帮助、管理压力、获取社交支持和与AI讨论健康方面更为自在，这表明AI有可能提供更广泛的情感支持，但也突显了防范有害使用的需求。总体而言，我们的探索性发现强调了负责任地开发面向消费者的情感福祉支持AI工具的重要性，同时赋予用户了解这些工具局限性的能力。 

---
# System of Agentic AI for the Discovery of Metal-Organic Frameworks 

**Title (ZH)**: 代理人工智能系统用于金属有机框架的发现 

**Authors**: Theo Jaffrelot Inizan, Sherry Yang, Aaron Kaplan, Yen-hsu Lin, Jian Yin, Saber Mirzaei, Mona Abdelgaid, Ali H. Alawadhi, KwangHwan Cho, Zhiling Zheng, Ekin Dogus Cubuk, Christian Borgs, Jennifer T. Chayes, Kristin A. Persson, Omar M. Yaghi  

**Link**: [PDF](https://arxiv.org/pdf/2504.14110)  

**Abstract**: Generative models and machine learning promise accelerated material discovery in MOFs for CO2 capture and water harvesting but face significant challenges navigating vast chemical spaces while ensuring synthetizability. Here, we present MOFGen, a system of Agentic AI comprising interconnected agents: a large language model that proposes novel MOF compositions, a diffusion model that generates crystal structures, quantum mechanical agents that optimize and filter candidates, and synthetic-feasibility agents guided by expert rules and machine learning. Trained on all experimentally reported MOFs and computational databases, MOFGen generated hundreds of thousands of novel MOF structures and synthesizable organic linkers. Our methodology was validated through high-throughput experiments and the successful synthesis of five "AI-dreamt" MOFs, representing a major step toward automated synthesizable material discovery. 

**Abstract (ZH)**: 基于Agentic AI的MOFGen系统：加速CO2捕获和水收集用MOFs材料发现但仍面临巨大挑战 

---
# Amplify Initiative: Building A Localized Data Platform for Globalized AI 

**Title (ZH)**: Amplify倡议：构建面向全球AI的本地化数据平台 

**Authors**: Qazi Mamunur Rashid, Erin van Liemt, Tiffany Shih, Amber Ebinama, Karla Barrios Ramos, Madhurima Maji, Aishwarya Verma, Charu Kalia, Jamila Smith-Loud, Joyce Nakatumba-Nabende, Rehema Baguma, Andrew Katumba, Chodrine Mutebi, Jagen Marvin, Eric Peter Wairagala, Mugizi Bruce, Peter Oketta, Lawrence Nderu, Obichi Obiajunwa, Abigail Oppong, Michael Zimba, Data Authors  

**Link**: [PDF](https://arxiv.org/pdf/2504.14105)  

**Abstract**: Current AI models often fail to account for local context and language, given the predominance of English and Western internet content in their training data. This hinders the global relevance, usefulness, and safety of these models as they gain more users around the globe. Amplify Initiative, a data platform and methodology, leverages expert communities to collect diverse, high-quality data to address the limitations of these models. The platform is designed to enable co-creation of datasets, provide access to high-quality multilingual datasets, and offer recognition to data authors. This paper presents the approach to co-creating datasets with domain experts (e.g., health workers, teachers) through a pilot conducted in Sub-Saharan Africa (Ghana, Kenya, Malawi, Nigeria, and Uganda). In partnership with local researchers situated in these countries, the pilot demonstrated an end-to-end approach to co-creating data with 155 experts in sensitive domains (e.g., physicians, bankers, anthropologists, human and civil rights advocates). This approach, implemented with an Android app, resulted in an annotated dataset of 8,091 adversarial queries in seven languages (e.g., Luganda, Swahili, Chichewa), capturing nuanced and contextual information related to key themes such as misinformation and public interest topics. This dataset in turn can be used to evaluate models for their safety and cultural relevance within the context of these languages. 

**Abstract (ZH)**: 当前的AI模型往往未能考虑到地方语境和语言，因为训练数据中占主导地位的是英语和西方互联网内容。这阻碍了这些模型在全球范围内的相关性、实用性和安全性，特别是在它们获得越来越多的全球用户之后。Amplify Initiative，一个数据平台和方法，通过利用专家社区收集多样性和高质量的数据来解决这些模型的局限性。该平台旨在促进与领域专家的合作数据集创建，提供高质量多语言数据集的访问权限，并为数据作者提供认可。本文介绍了一种通过在撒哈拉以南非洲地区（加纳、肯尼亚、马拉维、尼日利亚和乌干达）进行试点研究与领域专家（如医护人员、教师等）合作创建数据集的方法。与当地研究人员合作，试点展示了从155位专家（如医生、银行家、人类学家、人权倡导者）敏感领域中端到端合作创建数据集的方法。通过Android应用实施的方法，生成了一个包含8,091个 adversarial 查询的标注数据集，涉及七种语言（如卢干达语、斯瓦希里语、奇切瓦语），捕捉到与误导信息和公众兴趣主题相关的细微和情境信息。这个数据集可用于评估模型在这些语言背景下的安全性和文化相关性。 

---
# Coordinating Spinal and Limb Dynamics for Enhanced Sprawling Robot Mobility 

**Title (ZH)**: 增强 sprawling 机器人移动性的脊椎与肢体动力学协调 

**Authors**: Merve Atasever, Ali Okhovat, Azhang Nazaripouya, John Nisbet, Omer Kurkutlu, Jyotirmoy V. Deshmukh, Yasemin Ozkan Aydin  

**Link**: [PDF](https://arxiv.org/pdf/2504.14103)  

**Abstract**: Among vertebrates, salamanders, with their unique ability to transition between walking and swimming gaits, highlight the role of spinal mobility in locomotion. A flexible spine enables undulation of the body through a wavelike motion along the spine, aiding navigation over uneven terrains and obstacles. Yet environmental uncertainties, such as surface irregularities and variations in friction, can significantly disrupt body-limb coordination and cause discrepancies between predictions from mathematical models and real-world outcomes. Addressing this challenge requires the development of sophisticated control strategies capable of dynamically adapting to uncertain conditions while maintaining efficient locomotion. Deep reinforcement learning (DRL) offers a promising framework for handling non-deterministic environments and enabling robotic systems to adapt effectively and perform robustly under challenging conditions. In this study, we comparatively examine learning-based control strategies and biologically inspired gait design methods on a salamander-like robot. 

**Abstract (ZH)**: 在脊椎动物中，蝾螈因其独特的行走与游泳姿态转换能力，突显了脊柱灵活性在运动中的作用。灵活的脊柱通过脊柱上的波浪状运动实现身体的波浪式摆动，有助于在不平地形和障碍物上的导航。然而，环境不确定性，如表面不规则性和摩擦力变化，会显著干扰身体与肢体的协调，导致数学模型预测与实际结果之间存在差异。解决这一挑战需要开发出能够动态适应不确定条件并保持高效运动的复杂控制策略。深度强化学习（DRL）为处理非确定性环境并使机器人系统在挑战性条件下有效适应和稳健运行提供了有前景的框架。在这项研究中，我们比较了基于学习的控制策略和生物启发的步态设计方法在蝾螈类机器人上的应用。 

---
# 6G WavesFM: A Foundation Model for Sensing, Communication, and Localization 

**Title (ZH)**: 6G WavesFM：感测、通信与定位的础模型 

**Authors**: Ahmed Aboulfotouh, Elsayed Mohammed, Hatem Abou-Zeid  

**Link**: [PDF](https://arxiv.org/pdf/2504.14100)  

**Abstract**: This paper introduces WavesFM, a novel Wireless Foundation Model (WFM) framework, capable of supporting a wide array of communication, sensing, and localization tasks. Our proposed architecture combines a shared Vision Transformer (ViT) backbone with task-specific multi-layer perceptron (MLP) heads and incorporates Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning. This design promotes full parameter sharing across tasks, significantly reducing the computational and memory footprint without sacrificing performance. The model processes both image-like wireless modalities, such as spectrograms and channel state information (CSI), and in-phase and quadrature (IQ) signals arranged as orthogonal frequency-division multiplexing (OFDM) resource grids. We demonstrate the strong generalization capabilities of WavesFM through extensive experiments on four downstream tasks: Fifth Generation New Radio (5G NR) positioning; multiple-input multiple-output OFDM (MIMO-OFDM) channel estimation; human activity sensing; and radio-frequency (RF) signal classification. Compared to supervised baselines trained individually, our approach achieves superior performance while sharing 80% of its parameters across tasks. Furthermore, we show that pretraining on domain-relevant data not only boosts performance but also accelerates convergence, reducing training time by up to 5x. These results demonstrate that our unified WFM can support diverse tasks and deliver significant gains in both performance and efficiency, highlighting the transformative potential of foundation models to drive AI-native paradigms in future sixth-generation (6G) networks. 

**Abstract (ZH)**: 本文介绍了WavesFM，一种新型的无线基础模型（WFM）框架，支持广泛的通信、感测和定位任务。我们提出的设计结合了共享的Vision Transformer（ViT）骨干网络和特定任务的多层感知机（MLP）头部，并采用了低秩适应（LoRA）进行参数高效的微调。这一设计促进了跨任务的全参数共享，显著减少了计算和内存开销，同时不牺牲性能。该模型处理了包括频谱图和信道状态信息（CSI）在内的图像-like无线模态，以及按正交频分复用（OFDM）资源网格排列的同相和正交（IQ）信号。我们通过在四个下游任务上进行广泛的实验展示了WavesFM的强大泛化能力：第五代新型无线电（5G NR）定位；多输入多输出OFDM（MIMO-OFDM）信道估计；人体活动感测；射频（RF）信号分类。相较于单独训练的监督基线方法，我们的方法在共享80%参数的情况下实现了更优的性能。此外，我们还展示了在相关领域数据上的预训练不仅能提升性能，还能加速收敛，最多可减少5倍的训练时间。这些结果表明，我们的统一WFM可以支持多种任务，并在性能和效率方面取得显著提升，突显了基础模型在下一代（6G）网络中驱动AI原生范式的潜在变革性潜力。 

---
# Enhancing Math Learning in an LMS Using AI-Driven Question Recommendations 

**Title (ZH)**: 使用AI驱动的问题推荐增强LMS中的数学学习 

**Authors**: Justus Råmunddal  

**Link**: [PDF](https://arxiv.org/pdf/2504.14098)  

**Abstract**: This paper presents an AI-driven approach to enhance math learning in a modern Learning Management System (LMS) by recommending similar math questions. Deep embeddings for math questions are generated using Meta's Llama-3.2-11B-Vision-Instruct model, and three recommendation methods-cosine similarity, Self-Organizing Maps (SOM), and Gaussian Mixture Models (GMM)-are applied to identify similar questions. User interaction data, including session durations, response times, and correctness, are used to evaluate the methods. Our findings suggest that while cosine similarity produces nearly identical question matches, SOM yields higher user satisfaction whereas GMM generally underperforms, indicating that introducing variety to a certain degree may enhance engagement and thereby potential learning outcomes until variety is no longer balanced reasonably, which our data about the implementations of all three methods demonstrate. 

**Abstract (ZH)**: 基于AI驱动的方法在现代学习管理系统中通过推荐相似数学问题来增强数学学习 

---
# Leakage and Interpretability in Concept-Based Models 

**Title (ZH)**: 基于概念的模型中的泄漏与可解释性 

**Authors**: Enrico Parisini, Tapabrata Chakraborti, Chris Harbron, Ben D. MacArthur, Christopher R. S. Banerji  

**Link**: [PDF](https://arxiv.org/pdf/2504.14094)  

**Abstract**: Concept Bottleneck Models aim to improve interpretability by predicting high-level intermediate concepts, representing a promising approach for deployment in high-risk scenarios. However, they are known to suffer from information leakage, whereby models exploit unintended information encoded within the learned concepts. We introduce an information-theoretic framework to rigorously characterise and quantify leakage, and define two complementary measures: the concepts-task leakage (CTL) and interconcept leakage (ICL) scores. We show that these measures are strongly predictive of model behaviour under interventions and outperform existing alternatives in robustness and reliability. Using this framework, we identify the primary causes of leakage and provide strong evidence that Concept Embedding Models exhibit substantial leakage regardless of the hyperparameters choice. Finally, we propose practical guidelines for designing concept-based models to reduce leakage and ensure interpretability. 

**Abstract (ZH)**: 概念瓶颈模型旨在通过预测高层中间概念来提高可解释性，并被视为在高风险场景中部署的一种有前景的方法。然而，它们known to suffer from information leakage，即模型利用了在学习概念中编码的未预期信息。我们引入了一种信息论框架来严格地表征和量化这种泄漏，并定义了两种互补的度量标准：概念任务泄漏 (CTL) 分数和概念间泄漏 (ICL) 分数。我们证明了这些度量标准在干预下的模型行为预测能力强，并且在稳健性和可靠性方面优于现有替代方法。使用该框架，我们确定了泄漏的主要原因，并提供了强有力的证据，表明概念嵌入模型在任何超参数选择下都表现出显著的泄漏。最后，我们提出了实用指南，以减少泄漏并确保基于概念的模型的可解释性。 

---
# LogicTree: Structured Proof Exploration for Coherent and Rigorous Logical Reasoning with Large Language Models 

**Title (ZH)**: LogicTree：结构化证明探索以实现一致而严谨的逻辑推理 

**Authors**: Kang He, Kaushik Roy  

**Link**: [PDF](https://arxiv.org/pdf/2504.14089)  

**Abstract**: Large language models (LLMs) have achieved remarkable multi-step reasoning capabilities across various domains. However, LLMs still face distinct challenges in complex logical reasoning, as (1) proof-finding requires systematic exploration and the maintenance of logical coherence and (2) searching the right combination of premises at each reasoning step is inherently challenging in tasks with large premise space. To address this, we propose LogicTree, an inference-time modular framework employing algorithm-guided search to automate structured proof exploration and ensure logical coherence. Advancing beyond tree-of-thought (ToT), we incorporate caching mechanism into LogicTree to enable effective utilization of historical knowledge, preventing reasoning stagnation and minimizing redundancy. Furthermore, we address the combinatorial complexity of premise search by decomposing it into a linear process. The refined premise selection restricts subsequent inference to at most one derivation per step, enhancing reasoning granularity and enforcing strict step-by-step reasoning. Additionally, we introduce two LLM-free heuristics for premise prioritization, enabling strategic proof search. Experimental results on five datasets demonstrate that LogicTree optimally scales inference-time computation to achieve higher proof accuracy, surpassing chain-of-thought (CoT) and ToT with average gains of 23.6% and 12.5%, respectively, on GPT-4o. Moreover, within LogicTree, GPT-4o outperforms o3-mini by 7.6% on average. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各个领域已经实现了卓越的多步推理能力。然而，LLMs在复杂逻辑推理方面仍面临独特挑战，具体包括（1）证明发现需要系统性的探索和保持逻辑连贯性，（2）在大型前提空间中每一步推理找到正确前提的组合是固有的挑战。为了解决这一问题，我们提出了一种名为LogicTree的推理时模块化框架，通过算法指导的搜索自动化结构化的证明探索，并确保逻辑连贯性。超越Thought树（ToT），LogicTree引入了缓存机制，以有效利用历史知识，防止推理停滞并减少重复。此外，通过将其拆解为线性过程来应对前提搜索的组合复杂性。改进的前提选择将后续推理限制在每步最多一个演绎，增强推理精细度并强制执行严格的逐步推理。同时，我们引入了两种LLM-free启发式方法进行前提优先级排序，以实现战略性的证明搜索。在五个数据集上的实验结果表明，LogicTree能够最优地扩展推理解算时间，以实现更高的证明准确性，相对于CoT和ToT分别平均提高23.6%和12.5%，在GPT-4o上表现尤为突出；在LogicTree内部，GPT-4o相对于o3-mini平均提高了7.6%。 

---
# Evaluating Human-AI Interaction via Usability, User Experience and Acceptance Measures for MMM-C: A Creative AI System for Music Composition 

**Title (ZH)**: 基于MMM-C的音乐创作创意人工智能系统的人机交互评价：易用性、用户体验和接受度指标的研究 

**Authors**: Renaud Bougueng Tchemeube, Jeff Ens, Cale Plut, Philippe Pasquier, Maryam Safi, Yvan Grabit, Jean-Baptiste Rolland  

**Link**: [PDF](https://arxiv.org/pdf/2504.14071)  

**Abstract**: With the rise of artificial intelligence (AI), there has been increasing interest in human-AI co-creation in a variety of artistic domains including music as AI-driven systems are frequently able to generate human-competitive artifacts. Now, the implications of such systems for musical practice are being investigated. We report on a thorough evaluation of the user adoption of the Multi-Track Music Machine (MMM) as a co-creative AI tool for music composers. To do this, we integrate MMM into Cubase, a popular Digital Audio Workstation (DAW) by Steinberg, by producing a "1-parameter" plugin interface named MMM-Cubase (MMM-C), which enables human-AI co-composition. We contribute a methodological assemblage as a 3-part mixed method study measuring usability, user experience and technology acceptance of the system across two groups of expert-level composers: hobbyists and professionals. Results show positive usability and acceptance scores. Users report experiences of novelty, surprise and ease of use from using the system, and limitations on controllability and predictability of the interface when generating music. Findings indicate no significant difference between the two user groups. 

**Abstract (ZH)**: 随着人工智能（AI）的发展，对多种艺术领域包括音乐中的人机共创的兴趣日益增加，因为AI驱动的系统经常能够生成与人类竞争的艺术品。现在，正在研究此类系统对音乐实践的影响。我们报告了对多轨音乐机器（MMM）作为音乐作曲人协作型AI工具的用户采用进行全面评估的结果。为此，我们通过开发一个名为MMM-Cubase（MMM-C）的一参数插件接口，将MMM集成到Steinberg公司的流行数字音频工作站（DAW）Cubase中，从而实现人机协作创作。我们采用一种综合了三种混合方法的研究方法，从两个专家级作曲家群体：业余和专业作曲家的角度，测量系统的易用性、用户体验和技术接受度。结果显示积极的易用性和接纳性评分。用户报告了使用该系统时的新颖性、惊喜和易用性体验，同时也指出了生成音乐时接口的可控性和可预测性限制。研究发现，两个用户群体之间不存在显著差异。 

---
# A CMOS Probabilistic Computing Chip With In-situ hardware Aware Learning 

**Title (ZH)**: 一种具有就地硬件感知学习功能的CMOS概率计算芯片 

**Authors**: Jinesh Jhonsa, William Whitehead, David McCarthy, Shuvro Chowdhury, Kerem Camsari, Luke Theogarajan  

**Link**: [PDF](https://arxiv.org/pdf/2504.14070)  

**Abstract**: This paper demonstrates a probabilistic bit physics inspired solver with 440 spins configured in a Chimera graph, occupying an area of 0.44 mm^2. Area efficiency is maximized through a current-mode implementation of the neuron update circuit, standard cell design for analog blocks pitch-matched to digital blocks, and a shared power supply for both digital and analog components. Process variation related mismatches introduced by this approach are effectively mitigated using a hardware aware contrastive divergence algorithm during training. We validate the chip's ability to perform probabilistic computing tasks such as modeling logic gates and full adders, as well as optimization tasks such as MaxCut, demonstrating its potential for AI and machine learning applications. 

**Abstract (ZH)**: 基于概率比特物理的Chipira图模拟器设计与实现：440个自旋元件的面积效率最大化 

---
# Occlusion-Ordered Semantic Instance Segmentation 

**Title (ZH)**: 遮挡有序语义实例分割 

**Authors**: Soroosh Baselizadeh, Cheuk-To Yu, Olga Veksler, Yuri Boykov  

**Link**: [PDF](https://arxiv.org/pdf/2504.14054)  

**Abstract**: Standard semantic instance segmentation provides useful, but inherently 2D information from a single image. To enable 3D analysis, one usually integrates absolute monocular depth estimation with instance segmentation. However, monocular depth is a difficult task. Instead, we leverage a simpler single-image task, occlusion-based relative depth ordering, providing coarser but useful 3D information. We show that relative depth ordering works more reliably from occlusions than from absolute depth. We propose to solve the joint task of relative depth ordering and segmentation of instances based on occlusions. We call this task Occlusion-Ordered Semantic Instance Segmentation (OOSIS). We develop an approach to OOSIS that extracts instances and their occlusion order simultaneously from oriented occlusion boundaries and semantic segmentation. Unlike popular detect-and-segment framework for instance segmentation, combining occlusion ordering with instance segmentation allows a simple and clean formulation of OOSIS as a labeling problem. As a part of our solution for OOSIS, we develop a novel oriented occlusion boundaries approach that significantly outperforms prior work. We also develop a new joint OOSIS metric based both on instance mask accuracy and correctness of their occlusion order. We achieve better performance than strong baselines on KINS and COCOA datasets. 

**Abstract (ZH)**: 基于遮挡顺序的语义实例分割（OOSIS） 

---
# Sentiment Analysis of Airbnb Reviews: Exploring Their Impact on Acceptance Rates and Pricing Across Multiple U.S. Regions 

**Title (ZH)**: Airbnb评价的情感分析：探索其对多个美国地区接受率和定价的影响 

**Authors**: Ali Safari  

**Link**: [PDF](https://arxiv.org/pdf/2504.14053)  

**Abstract**: This research examines whether Airbnb guests' positive and negative comments influence acceptance rates and rental prices across six U.S. regions: Rhode Island, Broward County, Chicago, Dallas, San Diego, and Boston. Thousands of reviews were collected and analyzed using Natural Language Processing (NLP) to classify sentiments as positive or negative, followed by statistical testing (t-tests and basic correlations) on the average scores. The findings reveal that over 90 percent of reviews in each region are positive, indicating that having additional reviews does not significantly enhance prices. However, listings with predominantly positive feedback exhibit slightly higher acceptance rates, suggesting that sentiment polarity, rather than the sheer volume of reviews, is a more critical factor for host success. Additionally, budget listings often gather extensive reviews while maintaining competitive pricing, whereas premium listings sustain higher prices with fewer but highly positive reviews. These results underscore the importance of sentiment quality over quantity in shaping guest behavior and pricing strategies in an overwhelmingly positive review environment. 

**Abstract (ZH)**: 本研究考察了 Airbnb 客人正面和负面评论是否影响六大美地区域的接受率和租金价格：罗德岛、布劳沃德县、芝加哥、达拉斯、圣地亚哥和波士顿。收集并分析了数千条评论，使用自然语言处理（NLP）将情感分类为正面或负面，随后通过统计检验（t 检验和基本相关性分析）对平均得分进行分析。研究发现，每个地区的评论中有超过 90% 是正面的，表明额外的评论对提升价格影响不大。然而，主要正面反馈的房源展现出略微更高的接受率，表明情绪极性而不是评论数量是影响房东成功的关键因素。此外，经济型房源通常会积累大量评论同时保持竞争力价格，而高端房源则凭借少量但高度正面的评论维持较高价格。这些结果强调，在大量正面评论的环境中，情感质量比数量对客人行为和定价策略的影响更为重要。 

---
# A synthetic dataset of French electric load curves with temperature conditioning 

**Title (ZH)**: 带有温度条件的法电负荷曲线合成数据集 

**Authors**: Tahar Nabil, Ghislain Agoua, Pierre Cauchois, Anne De Moliner, Benoît Grossin  

**Link**: [PDF](https://arxiv.org/pdf/2504.14046)  

**Abstract**: The undergoing energy transition is causing behavioral changes in electricity use, e.g. with self-consumption of local generation, or flexibility services for demand control. To better understand these changes and the challenges they induce, accessing individual smart meter data is crucial. Yet this is personal data under the European GDPR. A widespread use of such data requires thus to create synthetic realistic and privacy-preserving samples. This paper introduces a new synthetic load curve dataset generated by conditional latent diffusion. We also provide the contracted power, time-of-use plan and local temperature used for generation. Fidelity, utility and privacy of the dataset are thoroughly evaluated, demonstrating its good quality and thereby supporting its interest for energy modeling applications. 

**Abstract (ZH)**: 正在进行的能源转型正在改变 electricity 使用行为，例如通过本地发电的自我消费或需求控制的柔性服务。为了更好地理解和应对这些变化及其带来的挑战，访问个体智能电表数据至关重要。然而，这些数据属于个人数据且受欧盟GDPR保护。因此，广泛使用这些数据需要创建合成的、具现实性和隐私保护的数据样本。本文介绍了一种由条件潜在扩散生成的新合成负荷曲线数据集，并提供了用于生成的功率合同、时间-of-use 计划和当地温度。对数据集的忠实度、效用和隐私进行了详尽评估，展示了其良好的质量，从而支持其在能源建模应用中的应用价值。 

---
# MEQA: A Meta-Evaluation Framework for Question & Answer LLM Benchmarks 

**Title (ZH)**: MEQA: 一种问答大语言模型基准的元评估框架 

**Authors**: Jaime Raldua Veuthey, Zainab Ali Majid, Suhas Hariharan, Jacob Haimes  

**Link**: [PDF](https://arxiv.org/pdf/2504.14039)  

**Abstract**: As Large Language Models (LLMs) advance, their potential for widespread societal impact grows simultaneously. Hence, rigorous LLM evaluations are both a technical necessity and social imperative. While numerous evaluation benchmarks have been developed, there remains a critical gap in meta-evaluation: effectively assessing benchmarks' quality. We propose MEQA, a framework for the meta-evaluation of question and answer (QA) benchmarks, to provide standardized assessments, quantifiable scores, and enable meaningful intra-benchmark comparisons. We demonstrate this approach on cybersecurity benchmarks, using human and LLM evaluators, highlighting the benchmarks' strengths and weaknesses. We motivate our choice of test domain by AI models' dual nature as powerful defensive tools and security threats. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的发展，其对社会的潜在影响也在扩大。因此，严格的LLM评估既是技术上的必要，也是社会的迫切需求。尽管已经开发了众多评估基准，但在元评估方面仍然存在关键缺口：即有效评估基准质量的方法。我们提出MEQA框架，用于元评估问答（QA）基准，以提供标准化评估、可量化的评分，并促进基准间的有意义比较。我们通过使用人类评估者和LLM评估者在网络安全基准上展示这一方法，突出各基准的优势和不足。我们选择这一测试领域的原因是AI模型兼具强大的防御工具和安全威胁双重性质。 

---
# Flowco: Rethinking Data Analysis in the Age of LLMs 

**Title (ZH)**: Flowco：重思LLM时代的数据分析 

**Authors**: Stephen N. Freund, Brooke Simon, Emery D. Berger, Eunice Jun  

**Link**: [PDF](https://arxiv.org/pdf/2504.14038)  

**Abstract**: Conducting data analysis typically involves authoring code to transform, visualize, analyze, and interpret data. Large language models (LLMs) are now capable of generating such code for simple, routine analyses. LLMs promise to democratize data science by enabling those with limited programming expertise to conduct data analyses, including in scientific research, business, and policymaking. However, analysts in many real-world settings must often exercise fine-grained control over specific analysis steps, verify intermediate results explicitly, and iteratively refine their analytical approaches. Such tasks present barriers to building robust and reproducible analyses using LLMs alone or even in conjunction with existing authoring tools (e.g., computational notebooks). This paper introduces Flowco, a new mixed-initiative system to address these challenges. Flowco leverages a visual dataflow programming model and integrates LLMs into every phase of the authoring process. A user study suggests that Flowco supports analysts, particularly those with less programming experience, in quickly authoring, debugging, and refining data analyses. 

**Abstract (ZH)**: 执行数据分析通常涉及编写代码以转换、可视化、分析和解释数据。大型语言模型（LLMs）现在能够为简单的常规分析生成此类代码。LLMs有望通过使那些编程经验有限的人能够进行数据分析来平民化数据科学，包括在科学研究、商业和政策制定中。然而，在许多实际应用场景中，分析师往往必须对特定分析步骤进行精细控制，显式验证中间结果，并迭代优化他们的分析方法。这些任务使得仅依赖LLMs或者与现有编写工具（如计算笔记本）结合使用时构建稳健且可重复的分析变得更加困难。本文介绍了Flowco，一种新的混合主动性系统，以应对这些挑战。Flowco利用了可视化数据流编程模型，并将LLMs整合到编写过程的每个阶段。用户研究显示，Flowco有助于分析师，特别是编程经验较少的分析师，快速编写、调试和优化数据分析。 

---
# LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models 

**Title (ZH)**: LoftUp: 基于坐标的学习特征上采样器用于视觉基础模型 

**Authors**: Haiwen Huang, Anpei Chen, Volodymyr Havrylov, Andreas Geiger, Dan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.14032)  

**Abstract**: Vision foundation models (VFMs) such as DINOv2 and CLIP have achieved impressive results on various downstream tasks, but their limited feature resolution hampers performance in applications requiring pixel-level understanding. Feature upsampling offers a promising direction to address this challenge. In this work, we identify two critical factors for enhancing feature upsampling: the upsampler architecture and the training objective. For the upsampler architecture, we introduce a coordinate-based cross-attention transformer that integrates the high-resolution images with coordinates and low-resolution VFM features to generate sharp, high-quality features. For the training objective, we propose constructing high-resolution pseudo-groundtruth features by leveraging class-agnostic masks and self-distillation. Our approach effectively captures fine-grained details and adapts flexibly to various input and feature resolutions. Through experiments, we demonstrate that our approach significantly outperforms existing feature upsampling techniques across various downstream tasks. Our code is released at this https URL. 

**Abstract (ZH)**: 基于视觉的础模型（VFMs）如DINOv2和CLIP已在各种下游任务中取得了 impressive 的成果，但它们有限的特征分辨率阻碍了在需要像素级理解的应用中的性能。特征上采样为解决这一挑战提供了有希望的方向。在本文中，我们确定了增强特征上采样的两个关键因素：上采样器架构和训练目标。对于上采样器架构，我们引入了一种基于坐标的交叉注意变换器，它可以将高分辨率图像与坐标和低分辨率的VFM特征结合，生成清晰的高质量特征。对于训练目标，我们提出通过利用类无感知掩模和自我蒸馏构建高分辨率伪 ground-truth 特征。我们的方法有效地捕捉到了细微的细节，并且能够灵活适应各种输入和特征分辨率。通过实验，我们展示了我们的方法在各种下游任务中显著优于现有特征上采样技术。我们的代码已发布在 this https URL。 

---
# Causal pieces: analysing and improving spiking neural networks piece by piece 

**Title (ZH)**: 因果性组件：逐个分析与提升脉冲神经网络 

**Authors**: Dominik Dold, Philipp Christian Petersen  

**Link**: [PDF](https://arxiv.org/pdf/2504.14015)  

**Abstract**: We introduce a novel concept for spiking neural networks (SNNs) derived from the idea of "linear pieces" used to analyse the expressiveness and trainability of artificial neural networks (ANNs). We prove that the input domain of SNNs decomposes into distinct causal regions where its output spike times are locally Lipschitz continuous with respect to the input spike times and network parameters. The number of such regions - which we call "causal pieces" - is a measure of the approximation capabilities of SNNs. In particular, we demonstrate in simulation that parameter initialisations which yield a high number of causal pieces on the training set strongly correlate with SNN training success. Moreover, we find that feedforward SNNs with purely positive weights exhibit a surprisingly high number of causal pieces, allowing them to achieve competitive performance levels on benchmark tasks. We believe that causal pieces are not only a powerful and principled tool for improving SNNs, but might also open up new ways of comparing SNNs and ANNs in the future. 

**Abstract (ZH)**: 我们介绍了一种源自“线性片段”思想的新型突触神经网络（SNN）概念，用于分析和训练人工神经网络（ANN）。我们证明SNN的输入域分解为不同的因果区域，在这些区域中，输出尖锋时间对输入尖锋时间和网络参数的局部Lipschitz连续。这样的区域数量——我们称为“因果片段”——是SNN逼近能力的度量。特别地，在模拟中我们证明，能够在训练集上产生高数量因果片段的参数初始化与SNN训练成功高度相关。此外，我们发现具有全正权重的前向SNN显示出惊人的高数量因果片段，使它们能够在基准任务上达到竞争力的性能水平。我们认为，因果片段不仅是改进SNN的一种强大且基于原理的工具，还可能在未来为比较SNN和ANN开辟新的途径。 

---
# Fashion-RAG: Multimodal Fashion Image Editing via Retrieval-Augmented Generation 

**Title (ZH)**: Fashion-RAG: 基于检索增强生成的多模态时尚图像编辑 

**Authors**: Fulvio Sanguigni, Davide Morelli, Marcella Cornia, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2504.14011)  

**Abstract**: In recent years, the fashion industry has increasingly adopted AI technologies to enhance customer experience, driven by the proliferation of e-commerce platforms and virtual applications. Among the various tasks, virtual try-on and multimodal fashion image editing -- which utilizes diverse input modalities such as text, garment sketches, and body poses -- have become a key area of research. Diffusion models have emerged as a leading approach for such generative tasks, offering superior image quality and diversity. However, most existing virtual try-on methods rely on having a specific garment input, which is often impractical in real-world scenarios where users may only provide textual specifications. To address this limitation, in this work we introduce Fashion Retrieval-Augmented Generation (Fashion-RAG), a novel method that enables the customization of fashion items based on user preferences provided in textual form. Our approach retrieves multiple garments that match the input specifications and generates a personalized image by incorporating attributes from the retrieved items. To achieve this, we employ textual inversion techniques, where retrieved garment images are projected into the textual embedding space of the Stable Diffusion text encoder, allowing seamless integration of retrieved elements into the generative process. Experimental results on the Dress Code dataset demonstrate that Fashion-RAG outperforms existing methods both qualitatively and quantitatively, effectively capturing fine-grained visual details from retrieved garments. To the best of our knowledge, this is the first work to introduce a retrieval-augmented generation approach specifically tailored for multimodal fashion image editing. 

**Abstract (ZH)**: 近年来，时尚行业 increasingly 采纳 AI 技术 以 提升 客户体验，受到电子商务平台和虚拟应用的普及驱动。在多种任务中，虚拟试衣和多模态服装图像编辑——利用文本、服装草图和身体姿态等多种输入模态——已成为研究的关键领域。扩散模型 已 成为 这种 生成任务 的 领导性 方法，提供 优越 的 图像质量 和 多样性。然而，现有大多数虚拟试衣方法 均 依赖 特定 的 服装输入，这在实际应用中往往并不可行，因为用户可能仅提供 文本 规格。为解决这一局限，本文提出 一种 新颖 方法——Fashion Retrieval-Augmented Generation (Fashion-RAG)，使用户可以基于文本形式提供的偏好定制服装项目。我们的方法检索多个与输入规格匹配的服装，并生成包含检索物品属性的个性化图像。为了实现这一点，我们采用 文本反转 技术，使检索出的服装图像能够投射到 Stable Diffusion 文本编码器 的 文本嵌入空间，从而使检索出的元素能够无缝集成到生成过程中。在 Dress Code 数据集上的实验结果表明，Fashion-RAG 在定性和定量上均优于现有方法，有效捕捉了检索服装的详细视觉细节。据我们所知，这是首 次 将检索增强生成 方法 专门 应用于多模态服装图像编辑的研究工作。 

---
# CPR: Leveraging LLMs for Topic and Phrase Suggestion to Facilitate Comprehensive Product Reviews 

**Title (ZH)**: CPR: 利用大语言模型进行主题和短语建议以促进全面的产品评价 

**Authors**: Ekta Gujral, Apurva Sinha, Lishi Ji, Bijayani Sanghamitra Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2504.13993)  

**Abstract**: Consumers often heavily rely on online product reviews, analyzing both quantitative ratings and textual descriptions to assess product quality. However, existing research hasn't adequately addressed how to systematically encourage the creation of comprehensive reviews that capture both customers sentiment and detailed product feature analysis. This paper presents CPR, a novel methodology that leverages the power of Large Language Models (LLMs) and Topic Modeling to guide users in crafting insightful and well-rounded reviews. Our approach employs a three-stage process: first, we present users with product-specific terms for rating; second, we generate targeted phrase suggestions based on these ratings; and third, we integrate user-written text through topic modeling, ensuring all key aspects are addressed. We evaluate CPR using text-to-text LLMs, comparing its performance against real-world customer reviews from Walmart. Our results demonstrate that CPR effectively identifies relevant product terms, even for new products lacking prior reviews, and provides sentiment-aligned phrase suggestions, saving users time and enhancing reviews quality. Quantitative analysis reveals a 12.3% improvement in BLEU score over baseline methods, further supported by manual evaluation of generated phrases. We conclude by discussing potential extensions and future research directions. 

**Abstract (ZH)**: 基于大型语言模型和主题建模的综合产品评论引导方法 

---
# PC-DeepNet: A GNSS Positioning Error Minimization Framework Using Permutation-Invariant Deep Neural Network 

**Title (ZH)**: PC-DeepNet：一种使用排列不变深度神经网络的GNSS定位误差最小化框架 

**Authors**: M. Humayun Kabir, Md. Ali Hasan, Md. Shafiqul Islam, Kyeongjun Ko, Wonjae Shin  

**Link**: [PDF](https://arxiv.org/pdf/2504.13990)  

**Abstract**: Global navigation satellite systems (GNSS) face significant challenges in urban and sub-urban areas due to non-line-of-sight (NLOS) propagation, multipath effects, and low received power levels, resulting in highly non-linear and non-Gaussian measurement error distributions. In light of this, conventional model-based positioning approaches, which rely on Gaussian error approximations, struggle to achieve precise localization under these conditions. To overcome these challenges, we put forth a novel learning-based framework, PC-DeepNet, that employs a permutation-invariant (PI) deep neural network (DNN) to estimate position corrections (PC). This approach is designed to ensure robustness against changes in the number and/or order of visible satellite measurements, a common issue in GNSS systems, while leveraging NLOS and multipath indicators as features to enhance positioning accuracy in challenging urban and sub-urban environments. To validate the performance of the proposed framework, we compare the positioning error with state-of-the-art model-based and learning-based positioning methods using two publicly available datasets. The results confirm that proposed PC-DeepNet achieves superior accuracy than existing model-based and learning-based methods while exhibiting lower computational complexity compared to previous learning-based approaches. 

**Abstract (ZH)**: 全球导航卫星系统（GNSS）在城市和亚城市区域面临着由非视距（NLOS）传播、多路径效应和接收到的信号功率低所导致的显著挑战，这导致了高度非线性和非高斯测量误差分布。鉴于此，依赖高斯误差近似的传统基于模型的定位方法在这些条件下难以实现精确定位。为克服这些挑战，我们提出了一种新型基于学习的框架PC-DeepNet，该框架采用不变置换（PI）深度神经网络（DNN）来估计位置校正（PC）。该方法旨在确保在可见卫星测量数量和/或顺序发生变化时的鲁棒性，这一问题是GNSS系统的常见问题，同时利用非视距和多路径指示器作为特征来提升在具有挑战性的城市和亚城市环境中的定位精度。为了验证所提出框架的性能，我们使用两个公开可用的数据集与最先进的基于模型和基于学习的定位方法进行定位误差比较。结果表明，所提出的PC-DeepNet在精度上优于现有基于模型和基于学习的方法，并且相比之前的基于学习的方法具有更低的计算复杂度。 

---
# Gradual Binary Search and Dimension Expansion : A general method for activation quantization in LLMs 

**Title (ZH)**: 渐进二分搜索与维度扩展：大规模语言模型中激活量化的一种通用方法 

**Authors**: Lucas Maisonnave, Cyril Moineau, Olivier Bichler, Fabrice Rastello  

**Link**: [PDF](https://arxiv.org/pdf/2504.13989)  

**Abstract**: Large language models (LLMs) have become pivotal in artificial intelligence, demonstrating strong capabilities in reasoning, understanding, and generating data. However, their deployment on edge devices is hindered by their substantial size, often reaching several billion parameters. Quantization is a widely used method to reduce memory usage and inference time, however LLMs present unique challenges due to the prevalence of outliers in their activations. In this work, we leverage the theoretical advantages of Hadamard matrices over random rotation matrices to push the boundaries of quantization in LLMs. We demonstrate that Hadamard matrices are more effective in reducing outliers, which are a significant obstacle in achieving low-bit quantization. Our method based on a gradual binary search enables 3-bit quantization for weights, activations, and key-value (KV) caches, resulting in a 40\% increase in accuracy on common benchmarks compared to SoTA methods. We extend the use of rotation matrices to support non-power-of-2 embedding dimensions, similar to the Qwen architecture, by employing the Paley algorithm. We theoretically demonstrates the superiority of Hadamard matrices in reducing this http URL achieved 3-bit quantization for weights, activations, and KV cache, significantly enhancing model performance. Our experimental results on multiple models family like Mistral, LLaMA, and Qwen demonstrate the effectiveness of our approach, outperforming existing methods and enabling practical 3-bit quantization. 

**Abstract (ZH)**: 基于 Hadamard 矩阵的大语言模型量化方法：超越传统方法实现高效低比特量化 

---
# Entropy Rectifying Guidance for Diffusion and Flow Models 

**Title (ZH)**: 熵校正指导下的扩散与流模型 

**Authors**: Tariq Berrada Ifriqi, Adriana Romero-Soriano, Michal Drozdzal, Jakob Verbeek, Karteek Alahari  

**Link**: [PDF](https://arxiv.org/pdf/2504.13987)  

**Abstract**: Guidance techniques are commonly used in diffusion and flow models to improve image quality and consistency for conditional generative tasks such as class-conditional and text-to-image generation. In particular, classifier-free guidance (CFG) -- the most widely adopted guidance technique -- contrasts conditional and unconditional predictions to improve the generated images. This results, however, in trade-offs across quality, diversity and consistency, improving some at the expense of others. While recent work has shown that it is possible to disentangle these factors to some extent, such methods come with an overhead of requiring an additional (weaker) model, or require more forward passes per sampling step. In this paper, we propose Entropy Rectifying Guidance (ERG), a simple and effective guidance mechanism based on inference-time changes in the attention mechanism of state-of-the-art diffusion transformer architectures, which allows for simultaneous improvements over image quality, diversity and prompt consistency. ERG is more general than CFG and similar guidance techniques, as it extends to unconditional sampling. ERG results in significant improvements in various generation tasks such as text-to-image, class-conditional and unconditional image generation. We also show that ERG can be seamlessly combined with other recent guidance methods such as CADS and APG, further boosting generation performance. 

**Abstract (ZH)**: 熵矫正引导（ERG）：一种基于推断时注意机制改变的简单有效引导机制 

---
# On the redundancy of short and heterogeneous sequences of belief revisions 

**Title (ZH)**: 关于信念修订的短异质序列的冗余性 

**Authors**: Paolo Liberatore  

**Link**: [PDF](https://arxiv.org/pdf/2504.13986)  

**Abstract**: Forgetting a specific belief revision episode may not erase information because the other revisions may provide the same information or allow to deduce it. Whether it does was proved coNP-hard for sequence of two arbitrary lexicographic revision or arbitrarily long lexicographic Horn revision. A polynomial algorithm is presented for the case of two Horn revision. Heterogeneous sequences of revisions were proved to belong in Delta2. Their previously proved coNP-hardness is enhanced by a proof of NP-hardness. 

**Abstract (ZH)**: 遗忘特定的信念修订事件未必会消除信息，因为其他修订可能会提供相同的信息或允许推导出该信息。对于任意两个字典序修订或任意长的字典序Horn修订序列，该问题已被证明为coNP-hard。对于两个Horn修订的情况，提出了一种多项式时间算法。不同类型的修订序列被证明属于Δ²。通过证明NP-hardness，增强并扩展了先前证明的coNP-hardness。 

---
# One Jump Is All You Need: Short-Cutting Transformers for Early Exit Prediction with One Jump to Fit All Exit Levels 

**Title (ZH)**: 一跳足以全身而退：用于早期退出预测的变压器捷径结构，以一跳适应所有退出层级 

**Authors**: Amrit Diggavi Seshadri  

**Link**: [PDF](https://arxiv.org/pdf/2504.13984)  

**Abstract**: To reduce the time and computational costs of inference of large language models, there has been interest in parameter-efficient low-rank early-exit casting of transformer hidden-representations to final-representations. Such low-rank short-cutting has been shown to outperform identity shortcuts at early model stages while offering parameter-efficiency in shortcut jumps. However, current low-rank methods maintain a separate early-exit shortcut jump to final-representations for each transformer intermediate block-level during inference. In this work, we propose selection of a single One-Jump-Fits-All (OJFA) low-rank shortcut that offers over a 30x reduction in shortcut parameter costs during inference. We show that despite this extreme reduction, our OJFA choice largely matches the performance of maintaining multiple shortcut jumps during inference and offers stable precision from all transformer block-levels for GPT2-XL, Phi3-Mini and Llama2-7B transformer models. 

**Abstract (ZH)**: 减少大型语言模型推理时间和计算成本的方法：提出一种单一的全局低秩快捷通道（OJFA）方法，以实现超过30倍的快捷通道参数成本减少，在GPT2-XL、Phi3-Mini和Llama2-7B变压器模型中提供稳定的精度。 

---
# CacheFormer: High Attention-Based Segment Caching 

**Title (ZH)**: CacheFormer：基于高注意力的 segment 缓存 

**Authors**: Sushant Singh, Ausif Mahmood  

**Link**: [PDF](https://arxiv.org/pdf/2504.13981)  

**Abstract**: Efficiently handling long contexts in transformer-based language models with low perplexity is an active area of research. Numerous recent approaches like Linformer, Longformer, Performer, and Structured state space models (SSMs)., have not fully resolved this problem. All these models strive to reduce the quadratic time complexity of the attention mechanism while minimizing the loss in quality due to the effective compression of the long context. Inspired by the cache and virtual memory principle in computers, where in case of a cache miss, not only the needed data is retrieved from the memory, but the adjacent data is also obtained, we apply this concept to handling long contexts by dividing it into small segments. In our design, we retrieve the nearby segments in an uncompressed form when high segment-level attention occurs at the compressed level. Our en-hancements for handling long context include aggregating four attention mechanisms consisting of short sliding window attention, long compressed segmented attention, dynamically retrieving top k high attention uncompressed segments, and overlapping segments in long segment attention to avoid segment fragmentation. These enhancements result in an architecture that outperforms ex-isting SOTA architectures with an average perplexity improvement of 8.5% over similar model sizes. 

**Abstract (ZH)**: 基于变压器的语言模型高效处理长上下文并保持低 perplexity 是一个活跃的研究领域。尽管 Linformer、Longformer、Performer 和结构化状态空间模型 (SSMs) 等众多近期方法有所尝试，但仍未完全解决这一问题。所有这些模型都致力于减轻注意力机制的二次时间复杂度，同时尽可能减少由于有效压缩长上下文而导致的质量损失。受计算机中的缓存和虚拟内存原理启发，当发生缓存缺失时，不仅会从内存中检索所需的数据，还会获取相邻的数据，我们在此原理基础上，通过将长上下文分割为小段，来处理长上下文。在我们的设计中，在压缩级别出现高段级注意力时，会以不压缩的形式检索附近的段。为了处理长上下文，我们增强了四种注意力机制，包括短滑动窗口注意力、长压缩分割注意力、动态检索高注意力不压缩段以及长段注意力中的重叠段，以避免段落碎片化。这些增强提升了架构性能，相较于同等模型规模，平均 perplexity 改进幅度为 8.5%。 

---
# Framework, Standards, Applications and Best practices of Responsible AI : A Comprehensive Survey 

**Title (ZH)**: 负责任人工智能的框架、标准、应用及最佳实践综述 

**Authors**: Thippa Reddy Gadekallu, Kapal Dev, Sunder Ali Khowaja, Weizheng Wang, Hailin Feng, Kai Fang, Sharnil Pandya, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13979)  

**Abstract**: Responsible Artificial Intelligence (RAI) is a combination of ethics associated with the usage of artificial intelligence aligned with the common and standard frameworks. This survey paper extensively discusses the global and national standards, applications of RAI, current technology and ongoing projects using RAI, and possible challenges in implementing and designing RAI in the industries and projects based on AI. Currently, ethical standards and implementation of RAI are decoupled which caters each industry to follow their own standards to use AI ethically. Many global firms and government organizations are taking necessary initiatives to design a common and standard framework. Social pressure and unethical way of using AI forces the RAI design rather than implementation. 

**Abstract (ZH)**: 负责任的人工智能（RAI）：伦理与全球及国家标准、应用、技术及挑战的研究 

---
# Gas Station of the Future: A Perspective on AI/ML and IoT in Retail Downstream 

**Title (ZH)**: 未来加油站：AI/ML与IoT在零售下游领域的视角 

**Authors**: Wrick Talukdar  

**Link**: [PDF](https://arxiv.org/pdf/2504.13976)  

**Abstract**: The gas station of the future is poised to transform from a simple fuel dispensing center into an intelligent retail hub, driven by advancements in Artificial Intelligence (AI), Machine Learning (ML), and the Internet of Things (IoT). This paper explores how technology is reshaping the retail downstream sector while briefly addressing the upstream and midstream segments. By leveraging AI/ML for predictive analytics, dynamic pricing, personalized customer engagement, and IoT for real-time monitoring and automation, the future gas station will redefine the fuel retail experience. Additionally, this paper incorporates statistics, AI/ML core technical concepts, mathematical formulations, case studies, and a proposed framework for a fully autonomous gas station. 

**Abstract (ZH)**: 未来加油站即将从简单的燃油供应中心转变为由人工智能、机器学习和物联网驱动的智能零售枢纽，论文探讨技术如何重塑零售下游产业，并简要涉及上游和中期产业。通过利用人工智能/机器学习进行预测分析、动态定价、个性化客户服务以及物联网进行实时监控和自动化，未来的加油站将重新定义燃油零售体验。此外，本文还包含统计数据、人工智能/机器学习核心技术概念、数学公式、案例研究及一个全自主加油站的拟议框架。 

---
# Multiscale Tensor Summation Factorization as a New Neural Network Layer (MTS Layer) for Multidimensional Data Processing 

**Title (ZH)**: 多尺度张量求和因子分解作为新型多维数据处理的神经网络层（MTS层） 

**Authors**: Mehmet Yamaç, Muhammad Numan Yousaf, Serkan Kiranyaz, Moncef Gabbouj  

**Link**: [PDF](https://arxiv.org/pdf/2504.13975)  

**Abstract**: Multilayer perceptrons (MLP), or fully connected artificial neural networks, are known for performing vector-matrix multiplications using learnable weight matrices; however, their practical application in many machine learning tasks, especially in computer vision, can be limited due to the high dimensionality of input-output pairs at each layer. To improve efficiency, convolutional operators have been utilized to facilitate weight sharing and local connections, yet they are constrained by limited receptive fields. In this paper, we introduce Multiscale Tensor Summation (MTS) Factorization, a novel neural network operator that implements tensor summation at multiple scales, where each tensor to be summed is obtained through Tucker-decomposition-like mode products. Unlike other tensor decomposition methods in the literature, MTS is not introduced as a network compression tool; instead, as a new backbone neural layer. MTS not only reduces the number of parameters required while enhancing the efficiency of weight optimization compared to traditional dense layers (i.e., unfactorized weight matrices in MLP layers), but it also demonstrates clear advantages over convolutional layers. The proof-of-concept experimental comparison of the proposed MTS networks with MLPs and Convolutional Neural Networks (CNNs) demonstrates their effectiveness across various tasks, such as classification, compression, and signal restoration. Additionally, when integrated with modern non-linear units such as the multi-head gate (MHG), also introduced in this study, the corresponding neural network, MTSNet, demonstrates a more favorable complexity-performance tradeoff compared to state-of-the-art transformers in various computer vision applications. The software implementation of the MTS layer and the corresponding MTS-based networks, MTSNets, is shared at this https URL. 

**Abstract (ZH)**: 多层张量求和因子化（MTS因子化）：一种多尺度张量求和的新神经网络运算符 

---
# Enhancing Stroke Diagnosis in the Brain Using a Weighted Deep Learning Approach 

**Title (ZH)**: 使用加权深度学习方法增强脑卒中诊断 

**Authors**: Yao Zhiwan, Reza Zarrab, Jean Dubois  

**Link**: [PDF](https://arxiv.org/pdf/2504.13974)  

**Abstract**: A brain stroke occurs when blood flow to a part of the brain is disrupted, leading to cell death. Traditional stroke diagnosis methods, such as CT scans and MRIs, are costly and time-consuming. This study proposes a weighted voting ensemble (WVE) machine learning model that combines predictions from classifiers like random forest, Deep Learning, and histogram-based gradient boosting to predict strokes more effectively. The model achieved 94.91% accuracy on a private dataset, enabling early risk assessment and prevention. Future research could explore optimization techniques to further enhance accuracy. 

**Abstract (ZH)**: 脑卒中发生时，脑部某部分的血液供应被中断，导致细胞死亡。传统的大脑中风诊断方法，如CT扫描和MRI，成本高且耗时。本研究提出了一种加权投票集成(WVE)机器学习模型，该模型结合了随机森林、深度学习和直方图基梯度提升等分类器的预测，以更有效地预测中风。该模型在私有数据集上实现了94.91%的准确率，能够进行早期风险评估和预防。未来的研究可以探索优化技术以进一步提高准确率。 

---
# Governance Challenges in Reinforcement Learning from Human Feedback: Evaluator Rationality and Reinforcement Stability 

**Title (ZH)**: 从人类反馈中增强学习的治理挑战：评价者的理性与增强稳定性 

**Authors**: Dana Alsagheer, Abdulrahman Kamal, Mohammad Kamal, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.13972)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is central in aligning large language models (LLMs) with human values and expectations. However, the process remains susceptible to governance challenges, including evaluator bias, inconsistency, and the unreliability of feedback. This study examines how the cognitive capacity of evaluators, specifically their level of rationality, affects the stability of reinforcement signals. A controlled experiment comparing high-rationality and low-rationality participants reveals that evaluators with higher rationality scores produce significantly more consistent and expert-aligned feedback. In contrast, lower-rationality participants demonstrate considerable variability in their reinforcement decisions ($p < 0.01$). To address these challenges and improve RLHF governance, we recommend implementing evaluator pre-screening, systematic auditing of feedback consistency, and reliability-weighted reinforcement aggregation. These measures enhance the fairness, transparency, and robustness of AI alignment pipelines. 

**Abstract (ZH)**: 人类反馈强化学习（RLHF）在对齐大型语言模型（LLMs）与人类价值观和期望中的作用至关重要。然而，这一过程仍易受治理挑战的影响，包括评价者偏见、不一致性和反馈的不可靠性。本研究探讨了评价者的认知能力，特别是其理性的水平，对强化信号稳定性的影响。对比高理性与低理性参与者的受控实验表明，高理性评分的评价者产生的反馈更加一致且更符合专家标准。相比之下，低理性参与者在强化决策方面表现出显著的不稳定性（$p < 0.01$）。为了应对这些挑战并改进RLHF的治理，我们建议实施评价者的预先筛选、反馈一致性的系统审计以及可靠性加权的强化聚合。这些措施增强了人工智能对齐管道的公平性、透明度和鲁棒性。 

---
# The Future of Internet of Things and Multimodal Language Models in 6G Networks: Opportunities and Challenges 

**Title (ZH)**: 6G网络中物联网与多模态语言模型的未来：机遇与挑战 

**Authors**: Abdelrahman Soliman  

**Link**: [PDF](https://arxiv.org/pdf/2504.13971)  

**Abstract**: Based on recent trends in artificial intelligence and IoT research. The cooperative potential of integrating the Internet of Things (IoT) and Multimodal Language Models (MLLMs) is presented in this survey paper for future 6G systems. It focuses on the applications of this integration in different fields, such as healthcare, agriculture, and smart cities, and investigates the four pillars of IoT integration, such as sensors, communication, processing, and security. The paper provides a comprehensive description of IoT and MLLM technologies and applications, addresses the role of multimodality in each pillar, and concludes with an overview of the most significant challenges and directions for future research. The general survey is a roadmap for researchers interested in tracing the application areas of MLLMs and IoT, highlighting the potential and challenges in this rapidly growing field. The survey recognizes the need to deal with data availability, computational expense, privacy, and real-time processing to harness the complete potential of IoT, MLLM, and 6G technology 

**Abstract (ZH)**: 基于人工智能和物联网研究的最新趋势，本文综述了将物联网(IoT)与多模态语言模型(MLLMs)集成的协同潜力，为未来的6G系统提供参考。本文集中探讨了这种集成在不同领域（如医疗保健、农业和智慧城市）的应用，并研究了物联网集成的四大支柱，即传感器、通信、处理和安全。文章全面描述了物联网和多模态语言模型的技术和应用，分析了在每个支柱中多模态的作用，并总结了这一快速发展的领域中最具挑战性的问题和未来研究方向。综述为有兴趣跟踪多模态语言模型和物联网应用领域的研究人员提供了一条 roadmap，并强调了利用物联网、多模态语言模型和6G技术全部潜力时所面临的机遇与挑战。 

---
# Tinker Tales: Interactive Storytelling Framework for Early Childhood Narrative Development and AI Literacy 

**Title (ZH)**: 玩转故事：面向幼儿叙事发展与人工智能 literacy 的交互式叙事框架 

**Authors**: Nayoung Choi, Peace Cyebukayire, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.13969)  

**Abstract**: This paper presents Tinker Tales, an interactive storytelling framework in the format of a board game, designed to support both narrative development and AI literacy in early childhood. The framework integrates tangible and speech-based interactions with AI through NFC chip-attached pawns and tokens, along with a speaker and microphone. Children select and define key story elements-such as characters, places, items, and emotions-using the pawns and tokens, providing further details to the AI and receiving proper assistance, similar to how adults prompt AI for specific tasks (e.g., writing). For evaluation, several game sessions were simulated with a child AI agent, and the quality and safety of the generated stories were assessed from various perspectives. This work highlights the potential of combining physical and digital elements in AI literacy, offering a safe and engaging way for children to learn how to effectively collaborate with AI. 

**Abstract (ZH)**: Tinker Tales：一种板游戏式互动叙事框架，支持儿童早期的叙事发展与AI素养 

---
# CONTINA: Confidence Interval for Traffic Demand Prediction with Coverage Guarantee 

**Title (ZH)**: CONTINA：带有覆盖保证的交通需求预测置信区间 

**Authors**: Chao Yang, Xiannan Huang, Shuhan Qiu, Yan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.13961)  

**Abstract**: Accurate short-term traffic demand prediction is critical for the operation of traffic systems. Besides point estimation, the confidence interval of the prediction is also of great importance. Many models for traffic operations, such as shared bike rebalancing and taxi dispatching, take into account the uncertainty of future demand and require confidence intervals as the input. However, existing methods for confidence interval modeling rely on strict assumptions, such as unchanging traffic patterns and correct model specifications, to guarantee enough coverage. Therefore, the confidence intervals provided could be invalid, especially in a changing traffic environment. To fill this gap, we propose an efficient method, CONTINA (Conformal Traffic Intervals with Adaptation) to provide interval predictions that can adapt to external changes. By collecting the errors of interval during deployment, the method can adjust the interval in the next step by widening it if the errors are too large or shortening it otherwise. Furthermore, we theoretically prove that the coverage of the confidence intervals provided by our method converges to the target coverage level. Experiments across four real-world datasets and prediction models demonstrate that the proposed method can provide valid confidence intervals with shorter lengths. Our method can help traffic management personnel develop a more reasonable and robust operation plan in practice. And we release the code, model and dataset in \href{ this https URL}{ Github}. 

**Abstract (ZH)**: 准确的短时交通需求预测对于交通系统运营至关重要。除了点估计外，预测的置信区间同样十分重要。许多交通运营模型，如共享自行车重新平衡和出租车调度，都会考虑未来需求的不确定性，并要求输入置信区间。然而，现有的置信区间建模方法依赖于严格的假设，如交通模式不变和模型规格正确，以保证足够的覆盖范围。因此，提供的置信区间可能是无效的，尤其是在不断变化的交通环境中。为此，我们提出了一种高效的 方法，CONTINA（Conformal Traffic Intervals with Adaptation），以提供能够适应外部变化的区间预测。通过收集部署期间的误差，该方法可以在下一次调整区间时，如果误差过大则使其变宽，否则使其变窄。此外，我们从理论上证明了由我们方法提供的置信区间的覆盖范围将收敛到目标覆盖水平。实验结果表明，本方法可以在较短的区间长度下提供有效的置信区间。该方法有助于交通管理人员在实践中制定更为合理和稳健的运营计划。我们已将代码、模型和数据集发布在GitHub上。 

---
# AI Safety Should Prioritize the Future of Work 

**Title (ZH)**: AI安全应关注工作未来 

**Authors**: Sanchaita Hazra, Bodhisattwa Prasad Majumder, Tuhin Chakrabarty  

**Link**: [PDF](https://arxiv.org/pdf/2504.13959)  

**Abstract**: Current efforts in AI safety prioritize filtering harmful content, preventing manipulation of human behavior, and eliminating existential risks in cybersecurity or biosecurity. While pressing, this narrow focus overlooks critical human-centric considerations that shape the long-term trajectory of a society. In this position paper, we identify the risks of overlooking the impact of AI on the future of work and recommend comprehensive transition support towards the evolution of meaningful labor with human agency. Through the lens of economic theories, we highlight the intertemporal impacts of AI on human livelihood and the structural changes in labor markets that exacerbate income inequality. Additionally, the closed-source approach of major stakeholders in AI development resembles rent-seeking behavior through exploiting resources, breeding mediocrity in creative labor, and monopolizing innovation. To address this, we argue in favor of a robust international copyright anatomy supported by implementing collective licensing that ensures fair compensation mechanisms for using data to train AI models. We strongly recommend a pro-worker framework of global AI governance to enhance shared prosperity and economic justice while reducing technical debt. 

**Abstract (ZH)**: 当前在AI安全性方面的努力主要集中在过滤有害内容、防止操纵人类行为以及消除网络安全或生物安全中的生存风险。虽然这些是紧迫的问题，但这种狭窄的焦点忽视了塑造社会长期轨迹的关键的人本因素。在本文中，我们识别了忽视AI对未来工作影响的风险，并建议全面的支持向有意义劳动演变的过渡，其中包含人类自主权。通过经济学理论的视角，我们强调了AI对未来生计的跨时期影响以及劳动市场结构变化对收入不平等的加剧。此外，主要AI开发利益相关者采取的封闭源代码方法类似于通过利用资源、在创造性劳动中培养平庸和垄断创新来寻求租金的行为。为此，我们主张建立一个基于集体许可的坚实国际版权体系，以确保公平的补偿机制，并训练AI模型使用数据。我们强烈建议建立一个有利于工人的全球AI治理体系，以促进共享繁荣和经济正义，同时减少技术债务。 

---
# ToolRL: Reward is All Tool Learning Needs 

**Title (ZH)**: ToolRL: 奖励即是工具学习所需的一切 

**Authors**: Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-Tür, Gokhan Tur, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2504.13958)  

**Abstract**: Current Large Language Models (LLMs) often undergo supervised fine-tuning (SFT) to acquire tool use capabilities. However, SFT struggles to generalize to unfamiliar or complex tool use scenarios. Recent advancements in reinforcement learning (RL), particularly with R1-like models, have demonstrated promising reasoning and generalization abilities. Yet, reward design for tool use presents unique challenges: multiple tools may be invoked with diverse parameters, and coarse-grained reward signals, such as answer matching, fail to offer the finegrained feedback required for effective learning. In this work, we present the first comprehensive study on reward design for tool selection and application tasks within the RL paradigm. We systematically explore a wide range of reward strategies, analyzing their types, scales, granularity, and temporal dynamics. Building on these insights, we propose a principled reward design tailored for tool use tasks and apply it to train LLMs using Group Relative Policy Optimization (GRPO). Empirical evaluations across diverse benchmarks demonstrate that our approach yields robust, scalable, and stable training, achieving a 17% improvement over base models and a 15% gain over SFT models. These results highlight the critical role of thoughtful reward design in enhancing the tool use capabilities and generalization performance of LLMs. All the codes are released to facilitate future research. 

**Abstract (ZH)**: 当前的大语言模型（LLMs）常常通过监督微调（SFT）来获得工具使用能力。然而，SFT在处理不熟悉或复杂的工具使用场景时表现不佳。最近在强化学习（RL）方面的进展，尤其是R1-like模型，展示出了有希望的推理和泛化能力。然而，工具使用任务的奖励设计面临着独特挑战：多个工具可能需要具有不同参数的调用，粗粒度的奖励信号，如答案匹配，无法提供有效学习所需的细粒度反馈。本文首次系统研究了在RL范式下工具选择和应用任务中的奖励设计。我们系统地探索了广泛的奖励策略，分析了它们的类型、规模、粒度和时间动态。基于这些洞察，我们提出了一套针对工具使用任务的原理性奖励设计方法，并将其应用于使用组相对策略优化（GRPO）训练LLMs。在多样基准上的实证评估表明，我们的方法能实现稳健、可扩展和稳定的学习，相对于基线模型提升了17%，相对于监督微调模型提升了15%。这些结果突出了精心设计的奖励在提升LLMs工具使用能力和泛化性能方面的重要作用。所有代码均已发布，以促进未来的研究。 

---
# Naming is framing: How cybersecurity's language problems are repeating in AI governance 

**Title (ZH)**: 命名即框架：网络安全语言问题如何在AI治理中重演 

**Authors**: Liane Potter  

**Link**: [PDF](https://arxiv.org/pdf/2504.13957)  

**Abstract**: Language is not neutral; it frames understanding, structures power, and shapes governance. This paper argues that misnomers like cybersecurity and artificial intelligence (AI) are more than semantic quirks; they carry significant governance risks by obscuring human agency, inflating expectations, and distorting accountability. Drawing on lessons from cybersecurity's linguistic pitfalls, such as the 'weakest link' narrative, this paper highlights how AI discourse is falling into similar traps with metaphors like 'alignment,' 'black box,' and 'hallucination.' These terms embed adversarial, mystifying, or overly technical assumptions into governance structures. In response, the paper advocates for a language-first approach to AI governance: one that interrogates dominant metaphors, foregrounds human roles, and co-develops a lexicon that is precise, inclusive, and reflexive. This paper contends that linguistic reform is not peripheral to governance but central to the construction of transparent, equitable, and anticipatory regulatory frameworks. 

**Abstract (ZH)**: 语言不是中性的；它塑造理解、结构权力并影响治理。本文认为，诸如网络安全和人工智能（AI）之类的术语不仅仅是语义上的怪癖；它们通过模糊人类agency、夸大期望和扭曲问责制带来了重要的治理风险。本文借鉴网络安全语言陷阱的经验教训，如“最弱环节”叙事，指出AI话语正在落入类似陷阱，使用诸如“对齐”、“黑箱”和“幻觉”之类的隐喻。这些术语将对抗性、迷惑性的或过度技术化的假设嵌入到治理结构中。为此，本文提倡在AI治理中采取语言先行的方法：这种方法质疑主导隐喻，强调人类的作用，并共同开发一个精确、包容和反思性的词汇表。本文认为，语言改革对于建构透明、公平和前瞻性的治理框架是核心而非边缘问题。 

---
# Thousand Voices of Trauma: A Large-Scale Synthetic Dataset for Modeling Prolonged Exposure Therapy Conversations 

**Title (ZH)**: 千声创伤：模型 prolonged exposure 治疗对话的大规模合成数据集 

**Authors**: Suhas BN, Dominik Mattioli, Saeed Abdullah, Rosa I. Arriaga, Chris W. Wiese, Andrew M. Sherrill  

**Link**: [PDF](https://arxiv.org/pdf/2504.13955)  

**Abstract**: The advancement of AI systems for mental health support is hindered by limited access to therapeutic conversation data, particularly for trauma treatment. We present Thousand Voices of Trauma, a synthetic benchmark dataset of 3,000 therapy conversations based on Prolonged Exposure therapy protocols for Post-traumatic Stress Disorder (PTSD). The dataset comprises 500 unique cases, each explored through six conversational perspectives that mirror the progression of therapy from initial anxiety to peak distress to emotional processing. We incorporated diverse demographic profiles (ages 18-80, M=49.3, 49.4% male, 44.4% female, 6.2% non-binary), 20 trauma types, and 10 trauma-related behaviors using deterministic and probabilistic generation methods. Analysis reveals realistic distributions of trauma types (witnessing violence 10.6%, bullying 10.2%) and symptoms (nightmares 23.4%, substance abuse 20.8%). Clinical experts validated the dataset's therapeutic fidelity, highlighting its emotional depth while suggesting refinements for greater authenticity. We also developed an emotional trajectory benchmark with standardized metrics for evaluating model responses. This privacy-preserving dataset addresses critical gaps in trauma-focused mental health data, offering a valuable resource for advancing both patient-facing applications and clinician training tools. 

**Abstract (ZH)**: 基于长期暴露疗法协议的大规模合成创伤治疗对话数据集： thousand voices of trauma 

---
# Generative System Dynamics in Recurrent Neural Networks 

**Title (ZH)**: 生成系统动力学在递归神经网络中的应用 

**Authors**: Michele Casoni, Tommaso Guidi, Alessandro Betti, Stefano Melacci, Marco Gori  

**Link**: [PDF](https://arxiv.org/pdf/2504.13951)  

**Abstract**: In this study, we investigate the continuous time dynamics of Recurrent Neural Networks (RNNs), focusing on systems with nonlinear activation functions. The objective of this work is to identify conditions under which RNNs exhibit perpetual oscillatory behavior, without converging to static fixed points. We establish that skew-symmetric weight matrices are fundamental to enable stable limit cycles in both linear and nonlinear configurations. We further demonstrate that hyperbolic tangent-like activation functions (odd, bounded, and continuous) preserve these oscillatory dynamics by ensuring motion invariants in state space. Numerical simulations showcase how nonlinear activation functions not only maintain limit cycles, but also enhance the numerical stability of the system integration process, mitigating those instabilities that are commonly associated with the forward Euler method. The experimental results of this analysis highlight practical considerations for designing neural architectures capable of capturing complex temporal dependencies, i.e., strategies for enhancing memorization skills in recurrent models. 

**Abstract (ZH)**: 本研究探讨了具有非线性激活函数的循环神经网络（RNN）的连续时间动力学，重点研究了在系统中实现持久振荡行为而不收敛于静态固定点的条件。我们证明了反对称权重矩阵是实现线性与非线性配置下稳定极限环的关键。进一步研究表明，类似双曲正切的激活函数（奇函数、有界且连续）通过在状态空间中保持运动不变量，来维护这些振荡动力学。数值模拟展示了非线性激活函数不仅能够维持极限环，还能增强系统积分过程的数值稳定性，减轻与向前欧拉方法相关的那些不稳定现象。本分析的实验结果强调了设计能够捕捉复杂时间依赖性的神经架构的实用考虑，即增强循环模型记忆能力的策略。 

---
# Open-Medical-R1: How to Choose Data for RLVR Training at Medicine Domain 

**Title (ZH)**: Open-Medical-R1: 如何在医疗领域选择数据进行RLVR训练 

**Authors**: Zhongxi Qiu, Zhang Zhang, Yan Hu, Heng Li, Jiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.13950)  

**Abstract**: This paper explores optimal data selection strategies for Reinforcement Learning with Verified Rewards (RLVR) training in the medical domain. While RLVR has shown exceptional potential for enhancing reasoning capabilities in large language models, most prior implementations have focused on mathematics and logical puzzles, with limited exploration of domain-specific applications like medicine. We investigate four distinct data sampling strategies from MedQA-USMLE: random sampling (baseline), and filtering using Phi-4, Gemma-3-27b-it, and Gemma-3-12b-it models. Using Gemma-3-12b-it as our base model and implementing Group Relative Policy Optimization (GRPO), we evaluate performance across multiple benchmarks including MMLU, GSM8K, MMLU-Pro, and CMMLU. Our findings demonstrate that models trained on filtered data generally outperform those trained on randomly selected samples. Notably, training on self-filtered samples (using Gemma-3-12b-it for filtering) achieved superior performance in medical domains but showed reduced robustness across different benchmarks, while filtering with larger models from the same series yielded better overall robustness. These results provide valuable insights into effective data organization strategies for RLVR in specialized domains and highlight the importance of thoughtful data selection in achieving optimal performance. You can access our repository (this https URL) to get the codes. 

**Abstract (ZH)**: 这篇论文探讨了在医学领域使用验证奖励强化学习（RLVR）训练时的最优数据选择策略。尽管RLVR在增强大型语言模型的推理能力方面表现出出色的潜力，但大多数之前的实现主要集中于数学和逻辑谜题，对医学等特定领域的应用探索有限。我们从MedQA-USMLE中研究了四种不同的数据采样策略：随机采样（基线）、以及使用Phi-4、Gemma-3-27b-it和Gemma-3-12b-it模型进行过滤。以Gemma-3-12b-it作为基础模型并采用组相对策略优化（GRPO），我们在MMLU、GSM8K、MMLU-Pro和CMMLU等多个基准上评估了性能。研究结果显示，使用过滤数据训练的模型通常优于随机选择样本训练的模型。值得注意的是，使用Gemma-3-12b-it进行自我过滤的样本在医学领域取得了更好的性能，但在不同基准上的鲁棒性较差，而使用该系列更大模型进行过滤则表现出更好的整体鲁棒性。这些结果为RLVR在专门领域的有效数据组织策略提供了宝贵见解，并强调了在实现最佳性能时进行精心数据选择的重要性。您可以访问我们的仓库（点击这里）获取代码。 

---
# On Revealing the Hidden Problem Structure in Real-World and Theoretical Problems Using Walsh Coefficient Influence 

**Title (ZH)**: 基于沃尔什系数影响在现实世界和理论问题中揭示隐藏问题结构的研究 

**Authors**: M. W. Przewozniczek, F. Chicano, R. Tinós, J. Nalepa, B. Ruszczak, A. M. Wijata  

**Link**: [PDF](https://arxiv.org/pdf/2504.13949)  

**Abstract**: Gray-box optimization employs Walsh decomposition to obtain non-linear variable dependencies and utilize them to propose masks of variables that have a joint non-linear influence on fitness value. These masks significantly improve the effectiveness of variation operators. In some problems, all variables are non-linearly dependent, making the aforementioned masks useless. We analyze the features of the real-world instances of such problems and show that many of their dependencies may have noise-like origins. Such noise-caused dependencies are irrelevant to the optimization process and can be ignored. To identify them, we propose extending the use of Walsh decomposition by measuring variable dependency strength that allows the construction of the weighted dynamic Variable Interaction Graph (wdVIG). wdVIGs adjust the dependency strength to mixed individuals. They allow the filtering of irrelevant dependencies and re-enable using dependency-based masks by variation operators. We verify the wdVIG potential on a large benchmark suite. For problems with noise, the wdVIG masks can improve the optimizer's effectiveness. If all dependencies are relevant for the optimization, i.e., the problem is not noised, the influence of wdVIG masks is similar to that of state-of-the-art structures of this kind. 

**Abstract (ZH)**: 灰盒优化采用沃尔什分解获取非线性变量依赖关系，并利用这些依赖关系提出具有联合非线性影响于适应值的变量掩码。这些掩码显著提高了变异操作的有效性。在某些问题中，所有变量都呈非线性依赖，使得上述掩码变得无用。我们分析了此类问题的实际实例特征，并显示其中许多依赖可能是噪声引起的。这些由噪声引起的依赖与优化过程无关，并且可以忽略。为了识别它们，我们建议通过测量变量依赖强度来扩展沃尔什分解的应用，从而构建加权动态变量交互图（wdVIG）。wdVIG 调整依赖强度以适应混合个体，并允许过滤无关依赖并重新启用基于依赖的掩码。我们通过大型基准套件验证了 wdVIG 的潜在价值。对于具有噪声的问题，wdVIG 掩码可以提高优化器的效果。如果所有依赖都对优化相关，即问题无噪声，wdVIG 掩码的影响与此类最先进的结构相当。 

---
# Using customized GPT to develop prompting proficiency in architectural AI-generated images 

**Title (ZH)**: 使用定制化GPT提升 architectural AI生成图像的提示技巧 

**Authors**: Juan David Salazar Rodriguez, Sam Conrad Joyce, Julfendi Julfendi  

**Link**: [PDF](https://arxiv.org/pdf/2504.13948)  

**Abstract**: This research investigates the use of customized GPT models to enhance prompting proficiency among architecture students when generating AI-driven images. Prompt engineering is increasingly essential in architectural education due to the widespread adoption of generative AI tools. This study utilized a mixed-methods experimental design involving architecture students divided into three distinct groups: a control group receiving no structured support, a second group provided with structured prompting guides, and a third group supported by both structured guides and interactive AI personas. Students engaged in reverse engineering tasks, first guessing provided image prompts and then generating their own prompts, aiming to boost critical thinking and prompting skills. Variables examined included time spent prompting, word count, prompt similarity, and concreteness. Quantitative analysis involved correlation assessments between these variables and a one-way ANOVA to evaluate differences across groups. While several correlations showed meaningful relationships, not all were statistically significant. ANOVA results indicated statistically significant improvements in word count, similarity, and concreteness, especially in the group supported by AI personas and structured prompting guides. Qualitative feedback complemented these findings, revealing enhanced confidence and critical thinking skills in students. These results suggest tailored GPT interactions substantially improve students' ability to communicate architectural concepts clearly and effectively. 

**Abstract (ZH)**: 本研究探讨了定制GPT模型在提升建筑学生生成AI驱动图像时的提示 proficiency 方面的应用。由于生成型AI工具的广泛应用，提示工程在建筑教育中变得越来越重要。本研究采用混合方法实验设计，将建筑学生分为三组：一组为对照组，未提供结构化支持；第二组提供了结构化提示指南；第三组则同时得到了结构化指南和互动AI角色的支持。学生进行了逆向工程任务，首先猜测提供的图像提示，然后生成自己的提示，旨在提升批判性思维和提示技巧。研究变量包括提示时间、字数、提示相似度和具体性。定量分析包括变量间的相关性评估和单因素方差分析（ANOVA）以评估组间差异。虽然一些相关性显示出有意义的关系，但并非所有都是统计显著的。ANOVA结果表明，在得到AI角色和结构化提示指南支持的组中，字数、相似度和具体性有统计显著的提升。定性反馈补充了这些发现，显示学生自信心和批判性思维能力有所提升。这些结果表明，个性化GPT交互显著提高了学生清晰有效地传达建筑概念的能力。 

---
# From job titles to jawlines: Using context voids to study generative AI systems 

**Title (ZH)**: 从职位名称到下巴线条：利用上下文空白研究生成式AI系统 

**Authors**: Shahan Ali Memon, Soham De, Sungha Kang, Riyan Mujtaba, Bedoor AlShebli, Katie Davis, Jaime Snyder, Jevin D. West  

**Link**: [PDF](https://arxiv.org/pdf/2504.13947)  

**Abstract**: In this paper, we introduce a speculative design methodology for studying the behavior of generative AI systems, framing design as a mode of inquiry. We propose bridging seemingly unrelated domains to generate intentional context voids, using these tasks as probes to elicit AI model behavior. We demonstrate this through a case study: probing the ChatGPT system (GPT-4 and DALL-E) to generate headshots from professional Curricula Vitae (CVs). In contrast to traditional ways, our approach assesses system behavior under conditions of radical uncertainty -- when forced to invent entire swaths of missing context -- revealing subtle stereotypes and value-laden assumptions. We qualitatively analyze how the system interprets identity and competence markers from CVs, translating them into visual portraits despite the missing context (i.e. physical descriptors). We show that within this context void, the AI system generates biased representations, potentially relying on stereotypical associations or blatant hallucinations. 

**Abstract (ZH)**: 本研究引入了一种 speculate 设计方法论以研究生成式 AI 系统的行为，将设计视为一种探究模式。我们提出将看似无关的领域进行对接以生成有意图的背景空白，并使用这些任务作为探针来激发 AI 模型的行为。我们通过案例研究进行了演示，对 ChatGPT 系统（GPT-4 和 DALL-E）进行探针测试，从专业简历生成头像。与传统方法不同，我们的方法在极端不确定性条件下评估系统行为——当被迫发明大量缺失的背景时——揭示了细微的刻板印象和价值取向的假设。我们定性分析了系统如何理解和转化简历中的身份和能力标志，并在缺乏背景下将其转化为视觉肖像（即，缺乏身体描述）。在这一背景空白中，AI 系统生成了有偏见的表示，可能依赖于刻板印象联想或显性的虚拟构建。 

---
# Evaluating Menu OCR and Translation: A Benchmark for Aligning Human and Automated Evaluations in Large Vision-Language Models 

**Title (ZH)**: 评估菜单OCR和翻译：大型视觉-语言模型中人工评估与自动化评估对齐的标准 

**Authors**: Zhanglin Wu, Tengfei Song, Ning Xie, Weidong Zhang, Mengli Zhu, Shuang Wu, Shiliang Sun, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13945)  

**Abstract**: The rapid advancement of large vision-language models (LVLMs) has significantly propelled applications in document understanding, particularly in optical character recognition (OCR) and multilingual translation. However, current evaluations of LVLMs, like the widely used OCRBench, mainly focus on verifying the correctness of their short-text responses and long-text responses with simple layout, while the evaluation of their ability to understand long texts with complex layout design is highly significant but largely overlooked. In this paper, we propose Menu OCR and Translation Benchmark (MOTBench), a specialized evaluation framework emphasizing the pivotal role of menu translation in cross-cultural communication. MOTBench requires LVLMs to accurately recognize and translate each dish, along with its price and unit items on a menu, providing a comprehensive assessment of their visual understanding and language processing capabilities. Our benchmark is comprised of a collection of Chinese and English menus, characterized by intricate layouts, a variety of fonts, and culturally specific elements across different languages, along with precise human annotations. Experiments show that our automatic evaluation results are highly consistent with professional human evaluation. We evaluate a range of publicly available state-of-the-art LVLMs, and through analyzing their output to identify the strengths and weaknesses in their performance, offering valuable insights to guide future advancements in LVLM development. MOTBench is available at this https URL. 

**Abstract (ZH)**: 大规模视觉语言模型的快速发展极大地推动了文档理解的应用，特别是在光学字符识别（OCR）和多语言翻译方面。然而，现有的大规模视觉语言模型评估，如广泛使用的OCRBench，主要集中在验证其短文本和简单布局长文本响应的正确性，而对其理解和翻译复杂布局长文本能力的评估则显得至关重要但被忽视。本文提出了一种专门的评估框架——菜单OCR和翻译基准（MOTBench），强调菜单翻译在跨文化沟通中的关键作用。MOTBench要求大规模视觉语言模型准确识别和翻译菜单上每道菜、价格以及单位项目，从而全面评估其视觉理解和语言处理能力。我们的基准数据集包含中文和英文菜单，布局复杂，字体多样，并且包含不同语言中的文化特定元素，同时还附有人工精确标注。实验结果显示，我们的自动评估结果与专业的人类评估高度一致。我们评估了多种公开的领先大规模视觉语言模型，并通过分析其输出来识别其性能的优点和不足，为未来的模型开发提供了有价值的见解。MOTBench可访问 [此链接]。 

---
# Mixer Metaphors: audio interfaces for non-musical applications 

**Title (ZH)**: 混合元喻：非音乐应用的音频接口 

**Authors**: Tace McNamara, Jon McCormack, Maria Teresa Llano  

**Link**: [PDF](https://arxiv.org/pdf/2504.13944)  

**Abstract**: The NIME conference traditionally focuses on interfaces for music and musical expression. In this paper we reverse this tradition to ask, can interfaces developed for music be successfully appropriated to non-musical applications? To help answer this question we designed and developed a new device, which uses interface metaphors borrowed from analogue synthesisers and audio mixing to physically control the intangible aspects of a Large Language Model. We compared two versions of the device, with and without the audio-inspired augmentations, with a group of artists who used each version over a one week period. Our results show that the use of audio-like controls afforded more immediate, direct and embodied control over the LLM, allowing users to creatively experiment and play with the device over its non-mixer counterpart. Our project demonstrates how cross-sensory metaphors can support creative thinking and embodied practice when designing new technological interfaces. 

**Abstract (ZH)**: NIME会议传统上专注于音乐和音乐表达的界面。本文我们逆转这一传统，提出问题：为音乐开发的界面能否成功应用于非音乐应用？为了回答这一问题，我们设计并开发了一种新设备，该设备采用了来自模拟合成器和音频混音的界面隐喻，用于物理控制大型语言模型的无形方面。我们对比了有和没有音频启发增强功能的两种设备版本，并在为期一周的时间内供一组艺术家使用。结果显示，使用类似音频的控制提供了对LLM更直接、更具体的控制，使用户能够在非混音版本上进行创意实验和玩弄设备。我们的项目展示了跨感官隐喻在设计新技术界面时如何支持创造性思考和身体实践。 

---
# Intelligence of Things: A Spatial Context-Aware Control System for Smart Devices 

**Title (ZH)**: 物联网中的智能：一种基于空间上下文的智能设备控制系統 

**Authors**: Sukanth Kalivarathan, Muhmmad Abrar Raja Mohamed, Aswathy Ravikumar, S Harini  

**Link**: [PDF](https://arxiv.org/pdf/2504.13942)  

**Abstract**: This paper introduces Intelligence of Things (INOT), a novel spatial context-aware control system that enhances smart home automation through intuitive spatial reasoning. Current smart home systems largely rely on device-specific identifiers, limiting user interaction to explicit naming conventions rather than natural spatial references. INOT addresses this limitation through a modular architecture that integrates Vision Language Models with IoT control systems to enable natural language commands with spatial context (e.g., "turn on the light near the window"). The system comprises key components including an Onboarding Inference Engine, Zero-Shot Device Detection, Spatial Topology Inference, and Intent-Based Command Synthesis. A comprehensive user study with 15 participants demonstrated INOT's significant advantages over conventional systems like Google Home Assistant, with users reporting reduced cognitive workload (NASA-TLX scores decreased by an average of 13.17 points), higher ease-of-use ratings, and stronger preference (14 out of 15 participants). By eliminating the need to memorize device identifiers and enabling context-aware spatial commands, INOT represents a significant advancement in creating more intuitive and accessible smart home control systems. 

**Abstract (ZH)**: 基于事物的智能（INOT）：一种通过直观的空间推理增强智能家居自动化的新颖空间感知控制系统 

---
# NEMOTRON-CROSSTHINK: Scaling Self-Learning beyond Math Reasoning 

**Title (ZH)**: NEMOTRON-CROSSTHINK：超越数学推理的自我学习扩展 

**Authors**: Syeda Nahida Akter, Shrimai Prabhumoye, Matvei Novikov, Seungju Han, Ying Lin, Evelina Bakhturi, Eric Nyberg, Yejin Choi, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2504.13941)  

**Abstract**: Large Language Models (LLMs) have shown strong reasoning capabilities, particularly when enhanced through Reinforcement Learning (RL). While prior work has successfully applied RL to mathematical reasoning -- where rules and correctness are well-defined -- generalizing these methods to broader reasoning domains remains challenging due to limited data, the lack of verifiable reward structures, and diverse task requirements. In this work, we propose NEMOTRON-CROSSTHINK, a framework that systematically incorporates multi-domain corpora, including both synthetic and real-world question-answer pairs, into RL training to improve generalization across diverse reasoning tasks. NEMOTRON-CROSSTHINK addresses key challenges by (1) incorporating data from varied sources spanning STEM, humanities, social sciences, etc.; (2) applying structured templates (e.g., multiple-choice and open-ended) to control answer-space complexity; (3) filtering for verifiable answers; and (4) optimizing data blending strategies that utilizes data from multiple sources effectively. Our approach enables scalable and verifiable reward modeling beyond mathematics and demonstrates improved accuracies on both math (MATH-500: +30.1%, AMC23:+27.5%) and non-math reasoning benchmarks (MMLU-PRO: +12.8%, GPQA-DIAMOND: +11.3%, AGIEVAL: +15.1%, SUPERGPQA: +3.8%). Moreover, NEMOTRON-CROSSTHINK exhibits significantly improved response efficiency -- using 28% fewer tokens for correct answers -- highlighting more focused and effective reasoning. Through NEMOTRON-CROSSTHINK, we demonstrate that integrating multi-domain, multi-format data in RL leads to more accurate, efficient, and generalizable LLMs. 

**Abstract (ZH)**: NEMOTRON-CROSSTHINK：一种通过多域数据增强的强化学习框架 

---
# Hashigo: A Next Generation Sketch Interactive System for Japanese Kanji 

**Title (ZH)**: Hashigo：下一代日语漢字交互绘图系统 

**Authors**: Paul Taele, Tracy Hammond  

**Link**: [PDF](https://arxiv.org/pdf/2504.13940)  

**Abstract**: Language students can increase their effectiveness in learning written Japanese by mastering the visual structure and written technique of Japanese kanji. Yet, existing kanji handwriting recognition systems do not assess the written technique sufficiently enough to discourage students from developing bad learning habits. In this paper, we describe our work on Hashigo, a kanji sketch interactive system which achieves human instructor-level critique and feedback on both the visual structure and written technique of students' sketched kanji. This type of automated critique and feedback allows students to target and correct specific deficiencies in their sketches that, if left untreated, are detrimental to effective long-term kanji learning. 

**Abstract (ZH)**: 语言学習者可以通过掌握日语漢字的視覺結構和書寫技巧來提高寫作日语的有效性。然而，現有的漢字書寫認知系統在評估書寫技巧方面并不充分，無法阻止學習者養成不良的學習習慣。本文介紹了我們對Hashigo的研究工作，这是一个漢字草稿交互系統，可以實現類似人類教練的評估和反饋，針對學生草書漢字的視覺結構和書寫技巧進行評価和指導。這種自動化的評估和反饋使學員能夠즉시纠正在草稿中出现的特定缺陷，這些缺陷如果不予治療，將對長期有效的漢字學習産生負面影響。 

---
# LLM-Driven NPCs: Cross-Platform Dialogue System for Games and Social Platforms 

**Title (ZH)**: 由LLM驱动的NPC：跨平台游戏和社会平台对话系统 

**Authors**: Li Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.13928)  

**Abstract**: NPCs in traditional games are often limited by static dialogue trees and a single platform for interaction. To overcome these constraints, this study presents a prototype system that enables large language model (LLM)-powered NPCs to communicate with players both in the game en vironment (Unity) and on a social platform (Discord). Dialogue logs are stored in a cloud database (LeanCloud), allowing the system to synchronize memory between platforms and keep conversa tions coherent. Our initial experiments show that cross-platform interaction is technically feasible and suggest a solid foundation for future developments such as emotional modeling and persistent memory support. 

**Abstract (ZH)**: 传统游戏中的NPC通常受限于静态对话树和单一的互动平台。为了突破这些限制，本研究提出了一种原型系统，该系统利用大型语言模型（LLM）使NPC能够在游戏环境中（Unity）和社交平台（Discord）上与玩家互动。对话日志存储在云数据库（LeanCloud）中，从而使系统能够在不同平台之间同步记忆并保持对话连贯。初步实验表明，跨平台互动在技术上是可行的，并为未来的发展，如情绪建模和持久记忆支持奠定了坚实的基础。 

---
# A Multi-Layered Research Framework for Human-Centered AI: Defining the Path to Explainability and Trust 

**Title (ZH)**: 以人为本的AI多层研究框架：通往可解释性和信任的道路 

**Authors**: Chameera De Silva, Thilina Halloluwa, Dhaval Vyas  

**Link**: [PDF](https://arxiv.org/pdf/2504.13926)  

**Abstract**: The integration of Artificial Intelligence (AI) into high-stakes domains such as healthcare, finance, and autonomous systems is often constrained by concerns over transparency, interpretability, and trust. While Human-Centered AI (HCAI) emphasizes alignment with human values, Explainable AI (XAI) enhances transparency by making AI decisions more understandable. However, the lack of a unified approach limits AI's effectiveness in critical decision-making scenarios. This paper presents a novel three-layered framework that bridges HCAI and XAI to establish a structured explainability paradigm. The framework comprises (1) a foundational AI model with built-in explainability mechanisms, (2) a human-centered explanation layer that tailors explanations based on cognitive load and user expertise, and (3) a dynamic feedback loop that refines explanations through real-time user interaction. The framework is evaluated across healthcare, finance, and software development, demonstrating its potential to enhance decision-making, regulatory compliance, and public trust. Our findings advance Human-Centered Explainable AI (HCXAI), fostering AI systems that are transparent, adaptable, and ethically aligned. 

**Abstract (ZH)**: 将人工智能集成到医疗、金融和自主系统等高 stakes 领域常常受限于透明度、可解释性和信任方面的担忧。以人为本的人工智能（HCAI）强调与人类价值观的契合，可解释的人工智能（XAI）通过使人工智能决策更具可理解性来增强透明度。然而，缺乏统一的方法限制了人工智能在关键决策场景中的有效性。本文提出了一种新颖的三层框架，将HCAI和XAI相结合，建立结构化的可解释性范式。该框架包括（1）具有内置可解释性机制的基础人工智能模型，（2）以人为本的解释层，根据认知负荷和用户专业知识定制解释，以及（3）通过实时用户交互优化解释的动态反馈循环。该框架在医疗、金融和软件开发等领域进行了评估，证明了其在增强决策、合规性和公众信任方面的潜力。我们的研究推进了以人为本的可解释人工智能（HCXAI），促进了透明、适应性强且伦理上一致的人工智能系统。 

---
# Modeling the quantum-like dynamics of human reliability ratings in Human-AI interactions by interaction dependent Hamiltonians 

**Title (ZH)**: 基于相互作用依赖哈密顿量的人类可靠性评级的量子似动态建模：人类-人工智能交互中的应用 

**Authors**: Johan van der Meer, Pamela Hoyte, Luisa Roeder, Peter Bruza  

**Link**: [PDF](https://arxiv.org/pdf/2504.13918)  

**Abstract**: As our information environments become ever more powered by artificial intelligence (AI), the phenomenon of trust in a human's interactions with this intelligence is becoming increasingly pertinent. For example, in the not too distant future, there will be teams of humans and intelligent robots involved in dealing with the repercussions of high-risk disaster situations such as hurricanes, earthquakes, or nuclear accidents. Even in such conditions of high uncertainty, humans and intelligent machines will need to engage in shared decision making, and trust is fundamental to the effectiveness of these interactions. A key challenge in modeling the dynamics of this trust is to provide a means to incorporate sensitivity to fluctuations in human trust judgments. In this article, we explore the ability of Quantum Random Walk models to model the dynamics of trust in human-AI interactions, and to integrate a sensitivity to fluctuations in participant trust judgments based on the nature of the interaction with the AI. We found that using empirical parameters to inform the use of different Hamiltonians can provide a promising means to model the evolution of trust in Human-AI interactions. 

**Abstract (ZH)**: 随着我们的信息环境日益依赖人工智能（AI），人类与这一智能互动中的信任现象变得 increasingly pertinent。例如，在不远的将来，人类和智能机器人将组成团队应对飓风、地震或核事故等高风险灾难情况的后果。即使在这种高度不确定的条件下，人类和智能机器也需要进行共享决策，而信任是这些互动有效性的基础。建模这种信任动态的一个关键挑战是提供一种方法，以敏感性地反映人类信任判断的变化。在本文中，我们探讨了量子随机游走模型在建模人类与AI互动中的信任动态方面的能力，并基于与AI互动的性质整合了参与者信任判断波动的敏感性。我们发现，使用实证参数来指导使用不同的哈密顿量是一种有前途的方法，用以建模人类与AI互动中信任的发展。 

---
# AI-Assisted Conversational Interviewing: Effects on Data Quality and User Experience 

**Title (ZH)**: AI辅助对话式访谈：对数据质量与用户体验的影响 

**Authors**: Soubhik Barari, Jarret Angbazo, Natalie Wang, Leah M. Christian, Elizabeth Dean, Zoe Slowinski, Brandon Sepulvado  

**Link**: [PDF](https://arxiv.org/pdf/2504.13908)  

**Abstract**: Standardized surveys scale efficiently but sacrifice depth, while conversational interviews improve response quality at the cost of scalability and consistency. This study bridges the gap between these methods by introducing a framework for AI-assisted conversational interviewing. To evaluate this framework, we conducted a web survey experiment where 1,800 participants were randomly assigned to text-based conversational AI agents, or "textbots", to dynamically probe respondents for elaboration and interactively code open-ended responses. We assessed textbot performance in terms of coding accuracy, response quality, and respondent experience. Our findings reveal that textbots perform moderately well in live coding even without survey-specific fine-tuning, despite slightly inflated false positive errors due to respondent acquiescence bias. Open-ended responses were more detailed and informative, but this came at a slight cost to respondent experience. Our findings highlight the feasibility of using AI methods to enhance open-ended data collection in web surveys. 

**Abstract (ZH)**: 标准化调查可以高效扩展但牺牲深度，而对话式访谈可以提高响应质量但牺牲扩展性和一致性。本研究通过引入AI辅助对话式访谈框架弥合了这两种方法之间的差距。为评估该框架，我们进行了一项网络调查实验，随机将1,800名参与者分配给基于文本的对话式AI代理，即“文本机器人”，以动态探询受访者并互动编码开放式回答。我们从编码准确性、响应质量和受访者体验三个方面评估了文本机器人的表现。研究结果显示，即使没有特定调查的微调，文本机器人在实时编码中表现适度良好，但由于受访者 acquiescence 偏差导致的轻微虚假积极错误有所增加。开放式回答更加详尽和信息丰富，但这对受访者的体验造成了一定的影响。我们的研究结果表明，使用AI方法增强网络调查中的开放式数据收集具有可行性。 

---
# Generative Framework for Personalized Persuasion: Inferring Causal, Counterfactual, and Latent Knowledge 

**Title (ZH)**: 个性化说服的生成框架：因果推理、反事实推理和潜在知识inferencing 

**Authors**: Donghuo Zeng, Roberto Legaspi, Yuewen Sun, Xinshuai Dong, Kazushi Ikeda, Peter Spirtes, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13904)  

**Abstract**: We hypothesize that optimal system responses emerge from adaptive strategies grounded in causal and counterfactual knowledge. Counterfactual inference allows us to create hypothetical scenarios to examine the effects of alternative system responses. We enhance this process through causal discovery, which identifies the strategies informed by the underlying causal structure that govern system behaviors. Moreover, we consider the psychological constructs and unobservable noises that might be influencing user-system interactions as latent factors. We show that these factors can be effectively estimated. We employ causal discovery to identify strategy-level causal relationships among user and system utterances, guiding the generation of personalized counterfactual dialogues. We model the user utterance strategies as causal factors, enabling system strategies to be treated as counterfactual actions. Furthermore, we optimize policies for selecting system responses based on counterfactual data. Our results using a real-world dataset on social good demonstrate significant improvements in persuasive system outcomes, with increased cumulative rewards validating the efficacy of causal discovery in guiding personalized counterfactual inference and optimizing dialogue policies for a persuasive dialogue system. 

**Abstract (ZH)**: 我们假设最优系统响应源自基于因果和反事实知识的适应性策略。反事实推理允许我们创建假设情景以检验替代系统响应的影响。我们通过因果发现这一过程加以增强，该过程识别出受底层因果结构指导的策略，以规范系统行为。此外，我们考虑可能影响用户-系统交互的心理构念和不可观测噪声作为潜在因素。我们证明了这些因素可以有效地进行估计。我们利用因果发现识别用户和系统声明中的策略级因果关系，指导个性化反事实对话的生成。我们将用户声明策略建模为因果因素，使系统策略能够被视为反事实行动。此外，我们基于反事实数据优化选择系统响应的策略。使用关于社会公益的现实世界数据集的实验结果表明，在具有增加累积奖励的情况下，因果发现能够显著改善说服系统的效果，并验证了其在引导个性化反事实推断和优化具有说服力对话系统对话策略方面的有效性。 

---
# Supporting Students' Reading and Cognition with AI 

**Title (ZH)**: 使用AI支持学生的阅读与认知 

**Authors**: Yue Fu, Alexis Hiniker  

**Link**: [PDF](https://arxiv.org/pdf/2504.13900)  

**Abstract**: With the rapid adoption of AI tools in learning contexts, it is vital to understand how these systems shape users' reading processes and cognitive engagement. We collected and analyzed text from 124 sessions with AI tools, in which students used these tools to support them as they read assigned readings for an undergraduate course. We categorized participants' prompts to AI according to Bloom's Taxonomy of educational objectives -- Remembering, Understanding, Applying, Analyzing, Evaluating. Our results show that ``Analyzing'' and ``Evaluating'' are more prevalent in users' second and third prompts within a single usage session, suggesting a shift toward higher-order thinking. However, in reviewing users' engagement with AI tools over several weeks, we found that users converge toward passive reading engagement over time. Based on these results, we propose design implications for future AI reading-support systems, including structured scaffolds for lower-level cognitive tasks (e.g., recalling terms) and proactive prompts that encourage higher-order thinking (e.g., analyzing, applying, evaluating). Additionally, we advocate for adaptive, human-in-the-loop features that allow students and instructors to tailor their reading experiences with AI, balancing efficiency with enriched cognitive engagement. Our paper expands the dialogue on integrating AI into academic reading, highlighting both its potential benefits and challenges. 

**Abstract (ZH)**: 随着AI工具在学习情境中的迅速采用，了解这些系统如何塑造用户的阅读过程和认知参与变得至关重要。我们收集并分析了124个使用AI工具的会话文本，这些学生在这些工具的支持下阅读了一门本科课程的指定读物。我们将用户对AI的提示按照布卢姆教育目标分类学进行分类——记忆、理解、应用、分析、评价。结果显示，“分析”和“评价”类型的提示在单次使用会话中的第二和第三个提示中更为常见，这表明了一种向高层次思考的转变。然而，在审查用户在数周内与AI工具的互动时，我们发现用户逐渐转向了被动的阅读参与。基于这些结果，我们提出了未来AI阅读支持系统的设想，包括为低层次认知任务提供结构化的支架（例如，回忆术语）以及促进高层次思考的主动提示（例如，分析、应用、评价）。此外，我们提倡具备适应性和人类在环功能的设计，让学生和教师能够根据需要调整他们的阅读体验，平衡效率与丰富的认知参与。我们的论文扩展了将AI整合到学术阅读中的对话，强调了其潜在的利弊。 

---
# Predicting Satisfaction of Counterfactual Explanations from Human Ratings of Explanatory Qualities 

**Title (ZH)**: 从解释质量的人类评价预测反事实解释的满意度 

**Authors**: Marharyta Domnich, Rasmus Moorits Veski, Julius Välja, Kadi Tulver, Raul Vicente  

**Link**: [PDF](https://arxiv.org/pdf/2504.13899)  

**Abstract**: Counterfactual explanations are a widely used approach in Explainable AI, offering actionable insights into decision-making by illustrating how small changes to input data can lead to different outcomes. Despite their importance, evaluating the quality of counterfactual explanations remains an open problem. Traditional quantitative metrics, such as sparsity or proximity, fail to fully account for human preferences in explanations, while user studies are insightful but not scalable. Moreover, relying only on a single overall satisfaction rating does not lead to a nuanced understanding of why certain explanations are effective or not. To address this, we analyze a dataset of counterfactual explanations that were evaluated by 206 human participants, who rated not only overall satisfaction but also seven explanatory criteria: feasibility, coherence, complexity, understandability, completeness, fairness, and trust. Modeling overall satisfaction as a function of these criteria, we find that feasibility (the actionability of suggested changes) and trust (the belief that the changes would lead to the desired outcome) consistently stand out as the strongest predictors of user satisfaction, though completeness also emerges as a meaningful contributor. Crucially, even excluding feasibility and trust, other metrics explain 58% of the variance, highlighting the importance of additional explanatory qualities. Complexity appears independent, suggesting more detailed explanations do not necessarily reduce satisfaction. Strong metric correlations imply a latent structure in how users judge quality, and demographic background significantly shapes ranking patterns. These insights inform the design of counterfactual algorithms that adapt explanatory qualities to user expertise and domain context. 

**Abstract (ZH)**: 基于行为的反事实解释质量评估：用户满意度与解释质量准则分析 

---
# The Human Robot Social Interaction (HSRI) Dataset: Benchmarking Foundational Models' Social Reasoning 

**Title (ZH)**: 人类机器人社会交互数据集：基础知识模型的社会推理基准测试 

**Authors**: Dong Won Lee, Yubin Kim, Denison Guvenoz, Sooyeon Jeong, Parker Malachowsky, Louis-Philippe Morency, Cynthia Breazeal, Hae Won Park  

**Link**: [PDF](https://arxiv.org/pdf/2504.13898)  

**Abstract**: Our work aims to advance the social reasoning of embodied artificial intelligence (AI) agents in real-world social interactions. Recently, language models (LMs) and foundational models (FMs) are being utilized as automatic evaluators of human-AI interactions with the goal of eventually being used to improve the policy of the AI agent. To enable further research in this direction, we introduce a large-scale real-world Human Robot Social Interaction (HSRI) Dataset to benchmark the capabilities of LMs and FMs to identify and reason about social interactions, specifically with regard to robot social errors and competencies . Our dataset consists of 400 real-world human social robot interaction videos and over 10K annotations, detailing the robot's social errors, competencies, rationale, and corrective actions, capturing unique aspects of human-AI interaction only present in real-world interactions. To further assess AI models' ability to reason about social interactions, we propose eight new benchmark tasks for evaluating centered around whether AI models can (1) evaluate social interactions via detecting social errors and competencies, (2) identify the explanatory factors associated to errors and competencies, (3) understand the flow of real-world social interactions, and (4) provide reasons and corrective actions for social errors. Human studies and experiments with modern LMs and FMs reveal that current models struggle with these tasks, demonstrating that our dataset and benchmark provides a step forward towards socially intelligent AI. 

**Abstract (ZH)**: 我们的工作旨在推动具身人工智能（AI）代理在现实世界社会互动中的社会推理能力。最近，语言模型（LMs）和基础模型（FMs）被用作人类-AI互动的自动评估器，旨在最终用于改进AI代理的政策。为了推动这一方向的进一步研究，我们引入了一个大规模的现实世界人类机器人社会互动（HSRI）数据集，用以评估LMs和FMs识别和推理社会互动的能力，特别是与机器人的社会错误和能力相关方面。该数据集包括400个真实世界的真人与社会机器人互动视频和超过10,000个注释，详细记录了机器人的社会错误、能力、推理和纠正措施，捕获了仅在现实世界互动中才存在的独特人类-AI交互方面。为了进一步评估AI模型在社会互动中的推理能力，我们提出了八个新的基准任务，围绕AI模型能否（1）通过检测社会错误和能力来评估社会互动，（2）识别与错误和能力相关的解释性因素，（3）理解现实世界社会互动的流程，（4）为社会错误提供理由和纠正措施。现代人类研究和实验表明，当前模型在这些任务中表现不佳，证明了我们的数据集和基准对于迈向社会智能AI的重要性。 

---
# Mozualization: Crafting Music and Visual Representation with Multimodal AI 

**Title (ZH)**: 模化：利用多模态AI创作音乐和视觉表现 

**Authors**: Wanfang Xu, Lixiang Zhao, Haiwen Song, Xinheng Song, Zhaolin Lu, Yu Liu, Min Chen, Eng Gee Lim, Lingyun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.13891)  

**Abstract**: In this work, we introduce Mozualization, a music generation and editing tool that creates multi-style embedded music by integrating diverse inputs, such as keywords, images, and sound clips (e.g., segments from various pieces of music or even a playful cat's meow). Our work is inspired by the ways people express their emotions -- writing mood-descriptive poems or articles, creating drawings with warm or cool tones, or listening to sad or uplifting music. Building on this concept, we developed a tool that transforms these emotional expressions into a cohesive and expressive song, allowing users to seamlessly incorporate their unique preferences and inspirations. To evaluate the tool and, more importantly, gather insights for its improvement, we conducted a user study involving nine music enthusiasts. The study assessed user experience, engagement, and the impact of interacting with and listening to the generated music. 

**Abstract (ZH)**: Mozualization：一种融合关键词、图像和音剪辑创作多风格嵌入音乐的工具及其用户研究 

---
# Maestoso: An Intelligent Educational Sketching Tool for Learning Music Theory 

**Title (ZH)**: Maestoso: 一种智能音乐理论绘图学习工具 

**Authors**: Paul Taele, Laura Barreto, Tracy Hammond  

**Link**: [PDF](https://arxiv.org/pdf/2504.13889)  

**Abstract**: Learning music theory not only has practical benefits for musicians to write, perform, understand, and express music better, but also for both non-musicians to improve critical thinking, math analytical skills, and music appreciation. However, current external tools applicable for learning music theory through writing when human instruction is unavailable are either limited in feedback, lacking a written modality, or assuming already strong familiarity of music theory concepts. In this paper, we describe Maestoso, an educational tool for novice learners to learn music theory through sketching practice of quizzed music structures. Maestoso first automatically recognizes students' sketched input of quizzed concepts, then relies on existing sketch and gesture recognition techniques to automatically recognize the input, and finally generates instructor-emulated feedback. From our evaluations, we demonstrate that Maestoso performs reasonably well on recognizing music structure elements and that novice students can comfortably grasp introductory music theory in a single session. 

**Abstract (ZH)**: 学习音乐理论不仅对音乐家提高作曲、表演、理解与表现音乐的技能具有实际益处，也对非音乐家提高批判性思维、数学分析能力和音乐鉴赏能力有益。然而，当前可用于在缺乏人类指导的情况下通过写作学习音乐理论的外部工具要么反馈有限，要么缺少书面表达模式，要么假定使用者对音乐理论概念已有较强的熟悉度。本文介绍了Maestoso，这是一种面向初学者的教育工具，通过练习测验过的音乐结构草图来学习音乐理论。Maestoso 首先自动识别学生草绘的测验概念输入，然后依赖现有的草图和手势识别技术自动识别输入，并最终生成类似教师的反馈。从我们的评估中可以看出，Maestoso 在识别音乐结构元素方面表现合理，且初学者可以在单次会话中舒适地掌握初步的音乐理论知识。 

---
# Kanji Workbook: A Writing-Based Intelligent Tutoring System for Learning Proper Japanese Kanji Writing Technique with Instructor-Emulated Assessment 

**Title (ZH)**: kanji 工作坊：一种基于书写的人工助手智能辅导系统，用于学习正确的日语汉字书写技巧并模拟教师评估 

**Authors**: Paul Taele, Jung In Koh, Tracy Hammond  

**Link**: [PDF](https://arxiv.org/pdf/2504.13888)  

**Abstract**: Kanji script writing is a skill that is often introduced to novice Japanese foreign language students for achieving Japanese writing mastery, but often poses difficulties to students with primarily English fluency due to their its vast differences with written English. Instructors often introduce various pedagogical methods -- such as visual structure and written techniques -- to assist students in kanji study, but may lack availability providing direct feedback on students' writing outside of class. Current educational applications are also limited due to lacking richer instructor-emulated feedback. We introduce Kanji Workbook, a writing-based intelligent tutoring system for students to receive intelligent assessment that emulates human instructor feedback. Our interface not only leverages students' computing devices for allowing them to learn, practice, and review the writing of prompted characters from their course's kanji script lessons, but also provides a diverse set of writing assessment metrics -- derived from instructor interviews and classroom observation insights -- through intelligent scoring and visual animations. We deployed our interface onto novice- and intermediate-level university courses over an entire academic year, and observed that interface users on average achieved higher course grades than their peers and also reacted positively to our interface's various features. 

**Abstract (ZH)**: 日文漢字书写是一种常常被外语日语初学者用于掌握日文书写的技能，但由于它与英文书写的巨大差异，往往给以英语为主导语言的学生带来困难。教师常常采用各种教学方法——如视觉结构和书写技术——来帮助学生学习汉字，但由于缺乏课后直接反馈，这些方法可能效果有限。当前的教育应用也受限于缺乏更丰富的人工模拟反馈。我们介绍了一款名为“汉字工作簿”的基于书写的人工智能辅导系统，以帮助学生获得模拟教师反馈的智能评估。我们的界面不仅利用学生的计算设备，让学生能够学习、练习和复习课程中所学的汉字书写，还提供了一组多样化的书写评估指标——这些指标是从教师访谈和课堂观察中提取的，通过智能评分和可视化动画实现。我们在整个学年中将该界面部署到了初学者和中级水平的大学课程中，并观察到使用该界面的学生平均获得了更高的课程成绩，同时也对该界面的各种功能表现出积极的反应。 

---
# Towards a Multimodal Document-grounded Conversational AI System for Education 

**Title (ZH)**: 面向教育领域的多模态文档 grounding 对话 AI 系统研究 

**Authors**: Karan Taneja, Anjali Singh, Ashok K. Goel  

**Link**: [PDF](https://arxiv.org/pdf/2504.13884)  

**Abstract**: Multimedia learning using text and images has been shown to improve learning outcomes compared to text-only instruction. But conversational AI systems in education predominantly rely on text-based interactions while multimodal conversations for multimedia learning remain unexplored. Moreover, deploying conversational AI in learning contexts requires grounding in reliable sources and verifiability to create trust. We present MuDoC, a Multimodal Document-grounded Conversational AI system based on GPT-4o, that leverages both text and visuals from documents to generate responses interleaved with text and images. Its interface allows verification of AI generated content through seamless navigation to the source. We compare MuDoC to a text-only system to explore differences in learner engagement, trust in AI system, and their performance on problem-solving tasks. Our findings indicate that both visuals and verifiability of content enhance learner engagement and foster trust; however, no significant impact in performance was observed. We draw upon theories from cognitive and learning sciences to interpret the findings and derive implications, and outline future directions for the development of multimodal conversational AI systems in education. 

**Abstract (ZH)**: 利用文本和图像的多媒体学习已被证明能 Compared to text-only instruction, multimedia learning using text and images has been shown to improve learning outcomes. 但教育领域的对话式AI系统主要依赖于基于文本的交互，而多媒体学习中的多模态对话尚未被探索。此外，将对话式AI应用于学习环境需要基于可靠的来源并具备可验证性以建立信任。我们提出了一种基于GPT-4o的多模态文档导向对话式AI系统MuDoC，该系统利用文档中的文本和视觉内容生成交织的文本和图像响应。其界面允许通过平滑导航到源内容来验证AI生成的内容。我们对比MuDoC与纯文本系统，探索学习者参与度、对AI系统的信任以及解决任务性能的差异。研究发现，视觉内容和内容的可验证性均能增强学习者的参与度并培养信任；但未观察到对性能的显著影响。我们结合认知科学和学习科学的理论来解释这些发现并推导出启示，并概述了在教育领域发展中多模态对话式AI系统未来的研究方向。 

---
# New care pathways for supporting transitional care from hospitals to home using AI and personalized digital assistance 

**Title (ZH)**: 利用AI和个人化数字辅助支持从医院向家庭过渡护理的新路径 

**Authors**: Ionut Anghel, Tudor Cioara, Roberta Bevilacqua, Federico Barbarossa, Terje Grimstad, Riitta Hellman, Arnor Solberg, Lars Thomas Boye, Ovidiu Anchidin, Ancuta Nemes, Camilla Gabrielsen  

**Link**: [PDF](https://arxiv.org/pdf/2504.13877)  

**Abstract**: Transitional care may play a vital role for the sustainability of Europe future healthcare system, offering solutions for relocating patient care from hospital to home therefore addressing the growing demand for medical care as the population is ageing. However, to be effective, it is essential to integrate innovative Information and Communications Technology technologies to ensure that patients with comorbidities experience a smooth and coordinated transition from hospitals or care centers to home, thereby reducing the risk of rehospitalization. In this paper, we present an overview of the integration of Internet of Things, artificial intelligence, and digital assistance technologies with traditional care pathways to address the challenges and needs of healthcare systems in Europe. We identify the current gaps in transitional care and define the technology mapping to enhance the care pathways, aiming to improve patient outcomes, safety, and quality of life avoiding hospital readmissions. Finally, we define the trial setup and evaluation methodology needed to provide clinical evidence that supports the positive impact of technology integration on patient care and discuss the potential effects on the healthcare system. 

**Abstract (ZH)**: 过渡期护理可能在欧洲未来医疗保健系统可持续性中发挥关键作用，通过将患者护理从医院转移到家庭，从而应对人口老龄化带来的日益增长的医疗服务需求。然而，为了有效实施，必须整合创新的信息化和通信技术，以确保共病患者能够顺利且协调地从医院或护理中心转移到家中，从而降低再次入院的风险。本文概述了将物联网、人工智能和数字辅助技术与传统护理路径结合以应对欧洲医疗保健系统面临的挑战和需求。我们确定了过渡期护理中的现有差距，并制定了技术规划以增强护理路径，旨在提高患者结果、安全性和生活质量，避免重新入院。最后，我们定义了试验设计和评估方法，以提供支持技术整合对患者护理产生积极影响的临床证据，并讨论了对医疗保健系统潜在影响。 

---
# Human aversion? Do AI Agents Judge Identity More Harshly Than Performance 

**Title (ZH)**: 人类的偏好？AI代理是否比绩效更严厉地评判身份。 

**Authors**: Yuanjun Feng, Vivek Chodhary, Yash Raj Shrestha  

**Link**: [PDF](https://arxiv.org/pdf/2504.13871)  

**Abstract**: This study examines the understudied role of algorithmic evaluation of human judgment in hybrid decision-making systems, a critical gap in management research. While extant literature focuses on human reluctance to follow algorithmic advice, we reverse the perspective by investigating how AI agents based on large language models (LLMs) assess and integrate human input. Our work addresses a pressing managerial constraint: firms barred from deploying LLMs directly due to privacy concerns can still leverage them as mediating tools (for instance, anonymized outputs or decision pipelines) to guide high-stakes choices like pricing or discounts without exposing proprietary data. Through a controlled prediction task, we analyze how an LLM-based AI agent weights human versus algorithmic predictions. We find that the AI system systematically discounts human advice, penalizing human errors more severely than algorithmic errors--a bias exacerbated when the agent's identity (human vs AI) is disclosed and the human is positioned second. These results reveal a disconnect between AI-generated trust metrics and the actual influence of human judgment, challenging assumptions about equitable human-AI collaboration. Our findings offer three key contributions. First, we identify a reverse algorithm aversion phenomenon, where AI agents undervalue human input despite comparable error rates. Second, we demonstrate how disclosure and positional bias interact to amplify this effect, with implications for system design. Third, we provide a framework for indirect LLM deployment that balances predictive power with data privacy. For practitioners, this research emphasize the need to audit AI weighting mechanisms, calibrate trust dynamics, and strategically design decision sequences in human-AI systems. 

**Abstract (ZH)**: 本研究探讨了算法评估人类判断在混合决策系统中的未充分研究的角色，这是管理研究中的一个关键空白。虽然现有文献集中于人类不愿意遵循算法建议，我们通过研究基于大规模语言模型（LLMs）的AI代理评估和整合人类输入的方式，逆转了这一视角。我们的工作解决了一个紧迫的管理限制：因隐私顾虑而不能直接部署LLMs的公司仍然可以通过将它们作为中介工具（例如，匿名输出或决策流程）来引导涉及价格或折扣等高风险选择，而无需暴露专有数据。通过一个受控的预测任务，我们分析了一个基于LLMs的AI代理如何权衡人类与算法预测。我们发现，该AI系统系统性地低估人类建议，对人类错误的惩罚比算法错误更为严厉，这一偏见在代理的身份（人类还是AI）被披露且人类处于第二位时更加严重。这些结果揭示了AI生成的信任指标与人类判断实际影响力之间的脱节，挑战了关于公平的人机合作的假设。我们的发现有三个主要贡献。首先，我们发现了一种反向算法厌恶现象，即尽管错误率相当，AI代理仍低估人类输入的价值。其次，我们展示了披露和位置偏见相互作用以放大此效应，这对系统设计有重要意义。最后，我们提供了一个框架，用于平衡预测能力和数据隐私的间接LLMs部署。对于实践者，本研究表明需要审查AI权重机制、校准信任动态，并在人机系统中战略性地设计决策序列。 

---
# Using Generative AI Personas Increases Collective Diversity in Human Ideation 

**Title (ZH)**: 使用生成式AI人格增加集体 ideation 多样性 

**Authors**: Yun Wan, Yoram M Kalman  

**Link**: [PDF](https://arxiv.org/pdf/2504.13868)  

**Abstract**: This study challenges the widely-reported tradeoff between generative AI's (GenAI) contribution to creative outcomes and decreased diversity of these outcomes. We modified the design of such a study, by Doshi and Hauser (2024), in which participants wrote short stories either aided or unaided by GenAI plot ideas[1]. In the modified study, plot ideas were generated through ten unique GenAI "personas" with diverse traits (e.g. cultural backgrounds, thinking styles, genre preferences), creating a pool of 300 story plots. While plot ideas from any individual persona showed high similarity (average cosine similarity of 0.92), ideas across different personas exhibited substantial variation (average similarity of 0.20). When human participants wrote stories based on these diverse plot ideas, their collective outputs maintained the same level of diversity as stories written without GenAI assistance, effectively eliminating the diversity reduction observed in [1]. Traditional text analytics further revealed that GenAI-assisted stories featured greater diversity in descriptive and emotional language compared to purely human-generated stories without GenAI assistance. Our findings demonstrate that introducing diversity at the AI input stage through distinct personas can preserve and potentially enhance the collective diversity of human creative outputs when collaborating with GenAI. 

**Abstract (ZH)**: 本研究挑战了广泛报道的生成式AI（GenAI）对创造性成果贡献与这些成果多样性降低之间的权衡关系。通过修改Doshi和Hauser（2024）的研究设计，参与者使用或未使用GenAI情节创意来撰写短篇故事。在修改后的研究中，通过十种具有不同特质的独特GenAI“角色”生成情节创意，创建了300个故事梗概池。虽然任何单一角色的情节创意显示出高相似性（平均余弦相似度为0.92），但不同角色之间的情节创意展现了显著的差异性（平均相似度为0.20）。当人类参与者基于这些多样性的情节创意撰写故事时，集体产出与未使用GenAI辅助撰写的故事多样性相当，有效消除了[Doshi和Hauser的研究]中观察到的多样性减少现象。传统文本分析进一步表明，使用GenAI辅助的故事在描述性和情感语言上显示出了比完全由人类生成的故事更多的多样性。我们的研究结果表明，在AI输入阶段通过不同角色引入多样性可以保留并可能增强与GenAI合作时人类创造性产出的集体多样性。 

---
# Skeleton-Based Transformer for Classification of Errors and Better Feedback in Low Back Pain Physical Rehabilitation Exercises 

**Title (ZH)**: 基于骨架的变压器在低背部疼痛物理康复练习中错误分类与更好反馈的研究 

**Authors**: Aleksa Marusic, Sao Mai Nguyen, Adriana Tapus  

**Link**: [PDF](https://arxiv.org/pdf/2504.13866)  

**Abstract**: Physical rehabilitation exercises suggested by healthcare professionals can help recovery from various musculoskeletal disorders and prevent re-injury. However, patients' engagement tends to decrease over time without direct supervision, which is why there is a need for an automated monitoring system. In recent years, there has been great progress in quality assessment of physical rehabilitation exercises. Most of them only provide a binary classification if the performance is correct or incorrect, and a few provide a continuous score. This information is not sufficient for patients to improve their performance. In this work, we propose an algorithm for error classification of rehabilitation exercises, thus making the first step toward more detailed feedback to patients. We focus on skeleton-based exercise assessment, which utilizes human pose estimation to evaluate motion. Inspired by recent algorithms for quality assessment during rehabilitation exercises, we propose a Transformer-based model for the described classification. Our model is inspired by the HyperFormer method for human action recognition, and adapted to our problem and dataset. The evaluation is done on the KERAAL dataset, as it is the only medical dataset with clear error labels for the exercises, and our model significantly surpasses state-of-the-art methods. Furthermore, we bridge the gap towards better feedback to the patients by presenting a way to calculate the importance of joints for each exercise. 

**Abstract (ZH)**: 自动监测系统建议的物理康复锻炼对于各种肌肉骨骼疾病的恢复和防止再次受伤有益。然而，在缺乏直接监督的情况下，患者参与度往往会随着时间的推移而下降，因此需要一个自动监测系统。近年来，在物理康复锻炼的质量评估方面取得了很大进展。大多数方法仅提供二元分类，即表现正确或错误，少数方法提供连续评分。这些信息对于患者提高表现并不充分。在这项工作中，我们提出了一种康复锻炼错误分类算法，从而迈出了为患者提供更详细反馈的第一步。我们重点关注基于骨架的锻炼评估，利用人体姿态估计来评估动作。受康复锻炼质量评估最新算法的启发，我们提出了一种基于Transformer的模型来实现描述的分类。我们的模型受到HyperFormer方法用于人类动作识别的启发，并针对我们的问题和数据集进行了调整。评估在KERAAL数据集上进行，因为它是唯一一个具有明确锻炼错误标签的医疗数据集，我们的模型显著超过了最先进的方法。此外，我们通过提出一种计算每个锻炼关节重要性的方法，进一步缩小了向患者提供更好反馈的差距。 

---
# A Survey on (M)LLM-Based GUI Agents 

**Title (ZH)**: 基于(M)LLM的GUI代理综述 

**Authors**: Fei Tang, Haolei Xu, Hang Zhang, Siqi Chen, Xingyu Wu, Yongliang Shen, Wenqi Zhang, Guiyang Hou, Zeqi Tan, Yuchen Yan, Kaitao Song, Jian Shao, Weiming Lu, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13865)  

**Abstract**: Graphical User Interface (GUI) Agents have emerged as a transformative paradigm in human-computer interaction, evolving from rule-based automation scripts to sophisticated AI-driven systems capable of understanding and executing complex interface operations. This survey provides a comprehensive examination of the rapidly advancing field of LLM-based GUI Agents, systematically analyzing their architectural foundations, technical components, and evaluation methodologies. We identify and analyze four fundamental components that constitute modern GUI Agents: (1) perception systems that integrate text-based parsing with multimodal understanding for comprehensive interface comprehension; (2) exploration mechanisms that construct and maintain knowledge bases through internal modeling, historical experience, and external information retrieval; (3) planning frameworks that leverage advanced reasoning methodologies for task decomposition and execution; and (4) interaction systems that manage action generation with robust safety controls. Through rigorous analysis of these components, we reveal how recent advances in large language models and multimodal learning have revolutionized GUI automation across desktop, mobile, and web platforms. We critically examine current evaluation frameworks, highlighting methodological limitations in existing benchmarks while proposing directions for standardization. This survey also identifies key technical challenges, including accurate element localization, effective knowledge retrieval, long-horizon planning, and safety-aware execution control, while outlining promising research directions for enhancing GUI Agents' capabilities. Our systematic review provides researchers and practitioners with a thorough understanding of the field's current state and offers insights into future developments in intelligent interface automation. 

**Abstract (ZH)**: 基于大语言模型的图形用户界面代理：架构、技术和评估方法综述 

---
# DoYouTrustAI: A Tool to Teach Students About AI Misinformation and Prompt Engineering 

**Title (ZH)**: 你信任AI吗：一个教学生了解AI错误信息和提示工程的工具 

**Authors**: Phillip Driscoll, Priyanka Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2504.13859)  

**Abstract**: AI, especially Large Language Models (LLMs) like ChatGPT, have rapidly developed and gained widespread adoption in the past five years, shifting user preference from traditional search engines. However, the generative nature of LLMs raises concerns about presenting misinformation as fact. To address this, we developed a web-based application that helps K-12 students enhance critical thinking by identifying misleading information in LLM responses about major historical figures. In this paper, we describe the implementation and design details of the DoYouTrustAI tool, which can be used to provide an interactive lesson which teaches students about the dangers of misinformation and how believable generative AI can make it seem. The DoYouTrustAI tool utilizes prompt engineering to present the user with AI generated summaries about the life of a historical figure. These summaries can be either accurate accounts of that persons life, or an intentionally misleading alteration of their history. The user is tasked with determining the validity of the statement without external resources. Our research questions for this work were:(RQ1) How can we design a tool that teaches students about the dangers of misleading information and of how misinformation can present itself in LLM responses? (RQ2) Can we present prompt engineering as a topic that is easily understandable for students? Our findings highlight the need to correct misleading information before users retain it. Our tool lets users select familiar individuals for testing to reduce random guessing and presents misinformation alongside known facts to maintain believability. It also provides pre-configured prompt instructions to show how different prompts affect AI responses. Together, these features create a controlled environment where users learn the importance of verifying AI responses and understanding prompt engineering. 

**Abstract (ZH)**: AI，尤其是大型语言模型（LLMs）如ChatGPT，在过去五年中迅速发展并得到广泛应用，改变了用户对传统搜索引擎的偏好。然而，生成式的特点使LLMs有可能呈现虚假信息作为事实。为应对这一问题，我们开发了一个基于Web的应用程序，帮助K-12学生通过识别大型语言模型关于重要历史人物响应中的误导信息来增强批判性思维能力。在本文中，我们描述了DoYouTrustAI工具的设计和实现细节，该工具可用于提供一个交互式的课程，向学生讲述虚假信息的危害以及生成式AI使其显得可信的方式。DoYouTrustAI工具通过提示工程向用户提供关于历史人物生平的AI生成摘要，这些摘要可以是准确的生平描述，也可以是故意篡改的历史记录。用户需要在没有外部资源的情况下判断陈述的真伪。本研究的研究问题是：（RQ1）我们如何设计一种工具来教育学生关于误导信息的危害以及虚假信息如何在LLM响应中呈现？（RQ2）我们能否将提示工程呈现为一个对学生易于理解的主题？我们的研究结果强调了在用户记住信息之前纠正误导信息的必要性。该工具允许用户选择熟悉的个体进行测试，以减少随机猜测，并将虚假信息与已知事实并置以保持可信度。此外，它还提供了预配置的提示指令，以展示不同提示如何影响AI响应。这些功能共同创造了一个受控环境，在这个环境中，用户可以学习验证AI响应的重要性并了解提示工程。 

---
# The Effect of Explainable AI-based Decision Support on Human Task Performance: A Meta-Analysis 

**Title (ZH)**: 基于可解释AI的决策支持对人类任务绩效的影响：一篇元分析 

**Authors**: Felix Haag  

**Link**: [PDF](https://arxiv.org/pdf/2504.13858)  

**Abstract**: The desirable properties of explanations in information systems have fueled the demands for transparency in artificial intelligence (AI) outputs. To address these demands, the field of explainable AI (XAI) has put forth methods that can support human decision-making by explaining AI outputs. However, current empirical works present inconsistent findings on whether such explanations help to improve users' task performance in decision support systems (DSS). In this paper, we conduct a meta-analysis to explore how XAI affects human performance in classification tasks. Our results show an improvement in task performance through XAI-based decision support, though explanations themselves are not the decisive driver for this improvement. The analysis reveals that the studies' risk of bias moderates the effect of explanations in AI, while the explanation type appears to play only a negligible role. Our findings contribute to the human computer interaction field by enhancing the understanding of human-XAI collaboration in DSS. 

**Abstract (ZH)**: 信息系统的解释特性推动了对人工智能输出透明度的需求。为了应对这一需求，可解释人工智能(XAI)领域提出了支持人类决策的方法，通过解释人工智能输出来辅助决策。然而，当前的实证研究在决策支持系统(DSS)中这些解释是否能改善用户任务性能方面结果不一。本文通过元分析探讨XAI如何影响人类在分类任务中的绩效。结果显示，基于XAI的决策支持可以提高任务绩效，但解释本身并不是这种改进的关键驱动因素。分析发现，研究的风险偏倚调节了解释在人工智能中的效果，而解释类型似乎起到了非常次要的作用。我们的研究结果在人机交互领域增强了对人类与XAI协作的理解。 

---
# Towards Balancing Preference and Performance through Adaptive Personalized Explainability 

**Title (ZH)**: 通过自适应个性化可解释性实现偏好与性能的平衡 

**Authors**: Andrew Silva, Pradyumna Tambwekar, Mariah Schrum, Matthew Gombolay  

**Link**: [PDF](https://arxiv.org/pdf/2504.13856)  

**Abstract**: As robots and digital assistants are deployed in the real world, these agents must be able to communicate their decision-making criteria to build trust, improve human-robot teaming, and enable collaboration. While the field of explainable artificial intelligence (xAI) has made great strides to enable such communication, these advances often assume that one xAI approach is ideally suited to each problem (e.g., decision trees to explain how to triage patients in an emergency or feature-importance maps to explain radiology reports). This fails to recognize that users have diverse experiences or preferences for interaction modalities. In this work, we present two user-studies set in a simulated autonomous vehicle (AV) domain. We investigate (1) population-level preferences for xAI and (2) personalization strategies for providing robot explanations. We find significant differences between xAI modes (language explanations, feature-importance maps, and decision trees) in both preference (p < 0.01) and performance (p < 0.05). We also observe that a participant's preferences do not always align with their performance, motivating our development of an adaptive personalization strategy to balance the two. We show that this strategy yields significant performance gains (p < 0.05), and we conclude with a discussion of our findings and implications for xAI in human-robot interactions. 

**Abstract (ZH)**: 随着机器人和数字助手在现实世界中的应用，这些代理必须能够沟通其决策标准以建立信任、提高人机协同作战能力并促进合作。虽然可解释人工智能（xAI）领域的研究已经取得了显著进展以实现这种沟通，这些进步往往假设每种问题都有一个最理想的方法（例如，使用决策树解释紧急情况下的病人分诊过程，或使用特征重要性图解释放射学报告）。这种方法未能认识到用户在交互方式上存在多样化的经验或偏好。在本研究中，我们在模拟的自动驾驶车辆（AV）领域进行了两项用户研究。我们探讨了（1）关于xAI的整体偏好以及（2）提供机器人解释的个性化策略。我们发现，在偏好（p < 0.01）和性能（p < 0.05）方面，xAI模式（语言解释、特征重要性图和决策树）之间存在显著差异。我们还观察到，参与者的态度与其表现并不总是相符，这促使我们开发了一种适应性的个性化策略来平衡两者。我们展示了这种方法在性能方面取得了显著成效（p < 0.05），并就我们发现的结果及其对人机交互中xAI的含义进行了讨论。 

---
# GenShin:geometry-enhanced structural graph embodies binding pose can better predicting compound-protein interaction affinity 

**Title (ZH)**: GenShin：几何增强的结构性蛋白质图更好地预测化合物-蛋白质相互作用亲和力 

**Authors**: Pingfei Zhu, Chenyang Zhao, Haishi Zhao, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.13853)  

**Abstract**: AI-powered drug discovery typically relies on the successful prediction of compound-protein interactions, which are pivotal for the evaluation of designed compound molecules in structure-based drug design and represent a core challenge in the field.
However, accurately predicting compound-protein affinity via regression models usually requires adequate-binding pose, which are derived from costly and complex experimental methods or time-consuming simulations with docking software. In response, we have introduced the GenShin model, which constructs a geometry-enhanced structural graph module that separately extracts additional features from proteins and compounds. Consequently, it attains an accuracy on par with mainstream models in predicting compound-protein affinities, while eliminating the need for adequate-binding pose as input. Our experimental findings demonstrate that the GenShin model vastly outperforms other models that rely on non-input docking conformations, achieving, or in some cases even exceeding, the performance of those requiring adequate-binding pose. Further experiments indicate that our GenShin model is more robust to inadequate-binding pose, affirming its higher suitability for real-world drug discovery scenarios. We hope our work will inspire more endeavors to bridge the gap between AI models and practical drug discovery challenges. 

**Abstract (ZH)**: AI赋能的药物发现通常依赖于成功预测化合物-蛋白质相互作用，这是基于结构的药物设计中评估设计的化合物分子的关键，也是该领域的核心挑战。然而，通过回归模型准确预测化合物-蛋白质亲和力通常需要充分结合的姿态，这些姿态是从昂贵且复杂的实验方法或耗时的对接软件模拟中获得的。为应对这一挑战，我们引入了GenShin模型，该模型构建了一个增强几何结构图模块，分别从蛋白质和化合物中提取额外特征。因此，它在预测化合物-蛋白质亲和力的准确性方面与主流模型相当，同时消除了对充分结合的姿态作为输入的需求。我们的实验结果表明，GenShin模型在依赖非输入对接构象的其他模型中表现远远优于后者，在某些情况下甚至超过了需要充分结合姿态的模型。进一步的实验表明，我们的GenShin模型在不充分结合姿态方面更具鲁棒性，证实了其在实际药物发现场景中的更高适用性。我们希望我们的工作能够激励更多努力缩小AI模型与实际药物发现挑战之间的差距。 

---
# From Interaction to Collaboration: How Hybrid Intelligence Enhances Chatbot Feedback 

**Title (ZH)**: 从交互到协作：混合智能如何增强聊天机器人反馈 

**Authors**: Janet Rafner, Ryan Q. Guloy, Eden W. Wen, Catherine M. Chiodo, Jacob Sherson  

**Link**: [PDF](https://arxiv.org/pdf/2504.13848)  

**Abstract**: Generative AI (GenAI) chatbots are becoming increasingly integrated into virtual assistant technologies, yet their success hinges on the ability to gather meaningful user feedback to improve interaction quality, system outcomes, and overall user acceptance. Successful chatbot interactions can enable organizations to build long-term relationships with their customers and users, supporting customer loyalty and furthering the organization's goals. This study explores the impact of two distinct narratives and feedback collection mechanisms on user engagement and feedback behavior: a standard AI-focused interaction versus a hybrid intelligence (HI) framed interaction. Initial findings indicate that while small-scale survey measures allowed for no significant differences in user willingness to leave feedback, use the system, or trust the system, participants exposed to the HI narrative statistically significantly provided more detailed feedback. These initial findings offer insights into designing effective feedback systems for GenAI virtual assistants, balancing user effort with system improvement potential. 

**Abstract (ZH)**: 生成式人工智能（GenAI）聊天机器人日益融入虚拟助手技术，但其成功取决于收集有意义的用户反馈的能力，以提高交互质量、系统效果和整体用户接受度。成功的聊天机器人交互可以帮助组织与其客户和用户建立长期关系，支持客户忠诚度并实现组织目标。本研究探讨了两种不同叙事和反馈收集机制对用户参与度和反馈行为的影响：标准AI导向的交互与混合智能（HI）框架的交互。初步结果显示，虽然小型调查措施未发现用户愿意留反馈、使用系统或信任系统的显著差异，但接受HI叙事的参与者在提供详细反馈方面显著更为活跃。这些初步结果为设计有效的GenAI虚拟助手反馈系统提供了见解，平衡了用户努力与系统改进潜力。 

---
