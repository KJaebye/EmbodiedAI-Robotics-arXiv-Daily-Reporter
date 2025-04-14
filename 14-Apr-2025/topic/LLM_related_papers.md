# Do LLMs trust AI regulation? Emerging behaviour of game-theoretic LLM agents 

**Title (ZH)**: Do LLMs遵守AI监管？博弈 theoretic LLM代理的新兴行为 

**Authors**: Alessio Buscemi, Daniele Proverbio, Paolo Bova, Nataliya Balabanova, Adeela Bashir, Theodor Cimpeanu, Henrique Correia da Fonseca, Manh Hong Duong, Elias Fernandez Domingos, Antonio M. Fernandes, Marcus Krellner, Ndidi Bianca Ogbo, Simon T. Powers, Fernando P. Santos, Zia Ush Shamszaman, Zhao Song, Alessandro Di Stefano, Anh Han  

**Link**: [PDF](https://arxiv.org/pdf/2504.08640)  

**Abstract**: There is general agreement that fostering trust and cooperation within the AI development ecosystem is essential to promote the adoption of trustworthy AI systems. By embedding Large Language Model (LLM) agents within an evolutionary game-theoretic framework, this paper investigates the complex interplay between AI developers, regulators and users, modelling their strategic choices under different regulatory scenarios. Evolutionary game theory (EGT) is used to quantitatively model the dilemmas faced by each actor, and LLMs provide additional degrees of complexity and nuances and enable repeated games and incorporation of personality traits. Our research identifies emerging behaviours of strategic AI agents, which tend to adopt more "pessimistic" (not trusting and defective) stances than pure game-theoretic agents. We observe that, in case of full trust by users, incentives are effective to promote effective regulation; however, conditional trust may deteriorate the "social pact". Establishing a virtuous feedback between users' trust and regulators' reputation thus appears to be key to nudge developers towards creating safe AI. However, the level at which this trust emerges may depend on the specific LLM used for testing. Our results thus provide guidance for AI regulation systems, and help predict the outcome of strategic LLM agents, should they be used to aid regulation itself. 

**Abstract (ZH)**: 促进可信人工智能系统采用的关键在于AI开发生态系统中信任与合作的培养：基于大型语言模型的进化博弈论框架下的战略行为研究 

---
# Task Memory Engine (TME): Enhancing State Awareness for Multi-Step LLM Agent Tasks 

**Title (ZH)**: 任务记忆引擎(TME): 提升多步LLM代理任务的状态意识 

**Authors**: Ye Ye  

**Link**: [PDF](https://arxiv.org/pdf/2504.08525)  

**Abstract**: Large Language Models (LLMs) are increasingly used as autonomous agents for multi-step tasks. However, most existing frameworks fail to maintain a structured understanding of the task state, often relying on linear prompt concatenation or shallow memory buffers. This leads to brittle performance, frequent hallucinations, and poor long-range coherence. In this work, we propose the Task Memory Engine (TME), a lightweight and structured memory module that tracks task execution using a hierarchical Task Memory Tree (TMT). Each node in the tree corresponds to a task step, storing relevant input, output, status, and sub-task relationships. We introduce a prompt synthesis method that dynamically generates LLM prompts based on the active node path, significantly improving execution consistency and contextual grounding. Through case studies and comparative experiments on multi-step agent tasks, we demonstrate that TME leads to better task completion accuracy and more interpretable behavior with minimal implementation overhead. The full implementation of TME is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地被用作执行多步骤任务的自主代理。然而，现有的大多数框架未能维持对任务状态的结构化理解，往往依赖于线性提示连接或浅层记忆缓冲。这导致了脆弱的表现、频繁的虚构叙述以及较差的长范围连贯性。在本文中，我们提出了一种轻量级且结构化的记忆模块Task Memory Engine（TME），该模块使用层次化的Task Memory Tree（TMT）跟踪任务执行。树中的每个节点对应于一个任务步骤，并存储相关信息、输出、状态以及子任务关系。我们引入了一种提示合成方法，能够根据当前活动节点路径动态生成LLM提示，显著提高了执行一致性并增强了上下文关联性。通过在多步骤代理任务上的案例研究和比较实验，我们证明了TME能够以最小的实现开销实现更好的任务完成准确性和更可解释的行为。TME的完整实现可在以下链接获取：this https URL。 

---
# MedRep: Medical Concept Representation for General Electronic Health Record Foundation Models 

**Title (ZH)**: 医_rep: 通用电子健康记录基础模型的医学概念表示 

**Authors**: Junmo Kim, Namkyeong Lee, Jiwon Kim, Kwangsoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.08329)  

**Abstract**: Electronic health record (EHR) foundation models have been an area ripe for exploration with their improved performance in various medical tasks. Despite the rapid advances, there exists a fundamental limitation: Processing unseen medical codes out of the vocabulary. This problem limits the generality of EHR foundation models and the integration of models trained with different vocabularies. To deal with this problem, we propose MedRep for EHR foundation models based on the observational medical outcome partnership (OMOP) common data model (CDM), providing the integrated medical concept representations and the basic data augmentation strategy for patient trajectories. For concept representation learning, we enrich the information of each concept with a minimal definition through large language model (LLM) prompts and enhance the text-based representations through graph ontology of OMOP vocabulary. Trajectory augmentation randomly replaces selected concepts with other similar concepts that have closely related representations to let the model practice with the concepts out-of-vocabulary. Finally, we demonstrate that EHR foundation models trained with MedRep better maintain the prediction performance in external datasets. Our code implementation is publicly available at this https URL. 

**Abstract (ZH)**: 基于OMOP通用数据模型的MedRep：用于电子健康记录基础模型的医疗概念表示和轨迹增强方法 

---
# Orchestrating Agents and Data for Enterprise: A Blueprint Architecture for Compound AI 

**Title (ZH)**: 企业中代理与数据的 orchestrating：复合人工智能的蓝图架构 

**Authors**: Eser Kandogan, Nikita Bhutani, Dan Zhang, Rafael Li Chen, Sairam Gurajada, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2504.08148)  

**Abstract**: Large language models (LLMs) have gained significant interest in industry due to their impressive capabilities across a wide range of tasks. However, the widespread adoption of LLMs presents several challenges, such as integration into existing applications and infrastructure, utilization of company proprietary data, models, and APIs, and meeting cost, quality, responsiveness, and other requirements. To address these challenges, there is a notable shift from monolithic models to compound AI systems, with the premise of more powerful, versatile, and reliable applications. However, progress thus far has been piecemeal, with proposals for agentic workflows, programming models, and extended LLM capabilities, without a clear vision of an overall architecture. In this paper, we propose a 'blueprint architecture' for compound AI systems for orchestrating agents and data for enterprise applications. In our proposed architecture the key orchestration concept is 'streams' to coordinate the flow of data and instructions among agents. Existing proprietary models and APIs in the enterprise are mapped to 'agents', defined in an 'agent registry' that serves agent metadata and learned representations for search and planning. Agents can utilize proprietary data through a 'data registry' that similarly registers enterprise data of various modalities. Tying it all together, data and task 'planners' break down, map, and optimize tasks and queries for given quality of service (QoS) requirements such as cost, accuracy, and latency. We illustrate an implementation of the architecture for a use-case in the HR domain and discuss opportunities and challenges for 'agentic AI' in the enterprise. 

**Abstract (ZH)**: 大规模语言模型（LLMs）因其实现广泛任务的出色能力而在行业中引起了广泛关注。然而，LLMs的广泛应用也带来了一些挑战，例如与现有应用程序和基础设施的集成、利用公司专有数据、模型和API，以及满足成本、质量、响应性及其他要求。为应对这些挑战，人们已经从单一模型转向复杂的AI系统，旨在构建更强大、多功能且更可靠的应用程序。然而，进展仍较为零散，仍缺乏清晰的整体架构愿景，包括代理工作流程、编程模型和扩展的LLM能力的提案。本文提出了一种“蓝本架构”，用于协调企业应用中的代理和数据。在我们的提议架构中，“流水线”是关键的协调概念，用于协调代理之间数据和指令的流动。“代理注册表”定义了代理的注册信息和学习表示，用于搜索和规划，现有企业中的专有模型和API被映射到“代理”。通过“数据注册表”注册各种模态的企业数据，代理可以利用这些数据。“数据和任务规划器”将任务和查询分解、映射和优化，以满足给定的服务质量（QoS）要求，如成本、准确性和延迟。我们展示了该架构在人力资源领域的一个应用场景，并讨论了企业环境中“自主AI”的机遇与挑战。 

---
# Visual Chronicles: Using Multimodal LLMs to Analyze Massive Collections of Images 

**Title (ZH)**: 视觉编年史：使用多模态大规模语言模型分析海量图像集 

**Authors**: Boyang Deng, Songyou Peng, Kyle Genova, Gordon Wetzstein, Noah Snavely, Leonidas Guibas, Thomas Funkhouser  

**Link**: [PDF](https://arxiv.org/pdf/2504.08727)  

**Abstract**: We present a system using Multimodal LLMs (MLLMs) to analyze a large database with tens of millions of images captured at different times, with the aim of discovering patterns in temporal changes. Specifically, we aim to capture frequent co-occurring changes ("trends") across a city over a certain period. Unlike previous visual analyses, our analysis answers open-ended queries (e.g., "what are the frequent types of changes in the city?") without any predetermined target subjects or training labels. These properties cast prior learning-based or unsupervised visual analysis tools unsuitable. We identify MLLMs as a novel tool for their open-ended semantic understanding capabilities. Yet, our datasets are four orders of magnitude too large for an MLLM to ingest as context. So we introduce a bottom-up procedure that decomposes the massive visual analysis problem into more tractable sub-problems. We carefully design MLLM-based solutions to each sub-problem. During experiments and ablation studies with our system, we find it significantly outperforms baselines and is able to discover interesting trends from images captured in large cities (e.g., "addition of outdoor dining,", "overpass was painted blue," etc.). See more results and interactive demos at this https URL. 

**Abstract (ZH)**: 我们提出了一种使用多模态大模型（MLLMs）分析包含数千万张在不同时间拍摄的图像的大数据库的系统，旨在发现时间变化中的模式。具体而言，我们旨在捕捉一定时期内某个城市中频繁共现的变化趋势（“趋势”）。与之前的视觉分析不同，我们的分析可以回答开放性查询（例如：“城市中频繁的变化类型是什么？”）而无需预先确定的目标主题或训练标签。这些特性使得先前的学习基于或无监督的视觉分析工具不再适用。我们标识出MLLMs作为一种新型工具，因其具有开放性的语义理解能力。然而，我们的数据集规模比MLLMs能够处理的上下文规模大四个数量级。因此，我们引入了一个自底向上的过程，将大规模的视觉分析问题分解为更易于管理的子问题。我们精心设计了基于MLLMs的解决方案来解决每个子问题。在系统实验和消融研究中，我们发现它显著优于基线，并能够从大城市拍摄的图像中发现有趣的趋势（例如：“户外用餐区增加”、“立交桥被漆成蓝色”等）。请访问此链接以查看更多结果和交互演示：见更多结果和交互演示请访问: https://xxxxx 

---
# Fast-Slow-Thinking: Complex Task Solving with Large Language Models 

**Title (ZH)**: 快慢思考：大规模语言模型解决复杂任务 

**Authors**: Yiliu Sun, Yanfang Zhang, Zicheng Zhao, Sheng Wan, Dacheng Tao, Chen Gong  

**Link**: [PDF](https://arxiv.org/pdf/2504.08690)  

**Abstract**: Nowadays, Large Language Models (LLMs) have been gradually employed to solve complex tasks. To face the challenge, task decomposition has become an effective way, which proposes to divide a complex task into multiple simpler subtasks and then solve them separately so that the difficulty of the original task can be reduced. However, the performance of existing task decomposition methods can be suboptimal when the task contains overly complex logic and constraints. In this situation, the solution generated by LLMs may deviate from the original purpose of the task, or contain redundant or even erroneous content. Therefore, inspired by the fact that humans possess two thinking systems including fast thinking and slow thinking, this paper introduces a new task decomposition method termed ``Fast-Slow-Thinking'' (FST), which stimulates LLMs to solve tasks through the cooperation of Fast Thinking (FT) and Slow Thinking (ST) steps. Here FT focuses more on the general and concise aspect of the task, and ST focuses more on the details of the task. In FT, LLMs are prompted to remove the constraints of the original task, therefore simplifying it to a general and concise one. In ST, we recall the constraints removed in FT, so that LLMs can improve the answer generated in FT to meet the requirements of the original task. Therefore, our FST method enables LLMs to consider a complex problem via a human-like cognition process from coarse to fine, the effectiveness of which has been well demonstrated by the experiments on three types of tasks. 

**Abstract (ZH)**: 如今，大型语言模型（LLMs）已被逐渐应用于解决复杂任务。为了应对这一挑战，任务分解已成为一种有效的方法，即将一个复杂的任务分解为多个简单的子任务，然后分别解决，从而降低原任务的难度。然而，当任务包含过于复杂的逻辑和约束时，现有的任务分解方法的表现可能不尽如人意。在这种情况下，LLMs生成的解决方案可能偏离任务的原始目标，或包含冗余甚至错误的内容。因此，受人类拥有快速思考和慢速思考两种思维方式的启发，本文提出了一种新的任务分解方法，称为“快速-慢速思考”（Fast-Slow-Thinking, FST），该方法通过快速思考（FT）和慢速思考（ST）步骤的协作促使LLMs解决任务。在FT中，LLMs被提示去除原任务的约束，从而使任务简化为一个一般和简洁的问题。在ST中，我们重新考虑在FT中去除的约束，从而使LLMs能够在满足原任务要求的基础上改进FT生成的答案。因此，我们的FST方法使LLMs能够通过从粗到细的人类认知过程来考虑复杂问题，其有效性已通过三种类型任务的实验得到验证。 

---
# Seaweed-7B: Cost-Effective Training of Video Generation Foundation Model 

**Title (ZH)**: 海藻-7B：视频生成基础模型的经济高效训练 

**Authors**: Team Seawead, Ceyuan Yang, Zhijie Lin, Yang Zhao, Shanchuan Lin, Zhibei Ma, Haoyuan Guo, Hao Chen, Lu Qi, Sen Wang, Feng Cheng, Feilong Zuo Xuejiao Zeng, Ziyan Yang, Fangyuan Kong, Zhiwu Qing, Fei Xiao, Meng Wei, Tuyen Hoang, Siyu Zhang, Peihao Zhu, Qi Zhao, Jiangqiao Yan, Liangke Gui, Sheng Bi, Jiashi Li, Yuxi Ren, Rui Wang, Huixia Li, Xuefeng Xiao, Shu Liu, Feng Ling, Heng Zhang, Houmin Wei, Huafeng Kuang, Jerry Duncan, Junda Zhang, Junru Zheng, Li Sun, Manlin Zhang, Renfei Sun, Xiaobin Zhuang, Xiaojie Li, Xin Xia, Xuyan Chi, Yanghua Peng, Yuping Wang, Yuxuan Wang, Zhongkai Zhao, Zhuo Chen, Zuquan Song, Zhenheng Yang, Jiashi Feng, Jianchao Yang, Lu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08685)  

**Abstract**: This technical report presents a cost-efficient strategy for training a video generation foundation model. We present a mid-sized research model with approximately 7 billion parameters (7B) called Seaweed-7B trained from scratch using 665,000 H100 GPU hours. Despite being trained with moderate computational resources, Seaweed-7B demonstrates highly competitive performance compared to contemporary video generation models of much larger size. Design choices are especially crucial in a resource-constrained setting. This technical report highlights the key design decisions that enhance the performance of the medium-sized diffusion model. Empirically, we make two observations: (1) Seaweed-7B achieves performance comparable to, or even surpasses, larger models trained on substantially greater GPU resources, and (2) our model, which exhibits strong generalization ability, can be effectively adapted across a wide range of downstream applications either by lightweight fine-tuning or continue training. See the project page at this https URL 

**Abstract (ZH)**: 本技术报告提出了一种成本高效的视频生成基础模型训练策略。我们介绍了使用约70亿参数（7B）的中期研究模型Seaweed-7B，该模型从零开始训练共使用了665,000个H100 GPU小时。即使使用了适度的计算资源，Seaweed-7B在与更大规模的 contemporaneous 视频生成模型相比时也展现了高度竞争力的性能。在资源受限的环境中，设计选择尤为关键。本技术报告强调了提升中期扩散模型性能的关键设计决策。实证研究表明：（1）Seaweed-7B在使用显著更多GPU资源训练的大型模型中表现出相当甚至更优的性能；（2）我们的模型表现出强大的泛化能力，可以通过轻量级微调或继续训练有效地适应广泛下游应用。请参见项目页面：[该链接] 

---
# Genius: A Generalizable and Purely Unsupervised Self-Training Framework For Advanced Reasoning 

**Title (ZH)**: 天才：一个通用且纯粹的无监督自我训练框架以进行高级推理 

**Authors**: Fangzhi Xu, Hang Yan, Chang Ma, Haiteng Zhao, Qiushi Sun, Kanzhi Cheng, Junxian He, Jun Liu, Zhiyong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08672)  

**Abstract**: Advancing LLM reasoning skills has captivated wide interest. However, current post-training techniques rely heavily on supervisory signals, such as outcome supervision or auxiliary reward models, which face the problem of scalability and high annotation costs. This motivates us to enhance LLM reasoning without the need for external supervision. We introduce a generalizable and purely unsupervised self-training framework, named Genius. Without external auxiliary, Genius requires to seek the optimal response sequence in a stepwise manner and optimize the LLM. To explore the potential steps and exploit the optimal ones, Genius introduces a stepwise foresight re-sampling strategy to sample and estimate the step value by simulating future outcomes. Further, we recognize that the unsupervised setting inevitably induces the intrinsic noise and uncertainty. To provide a robust optimization, we propose an advantage-calibrated optimization (ACO) loss function to mitigate estimation inconsistencies. Combining these techniques together, Genius provides an advanced initial step towards self-improve LLM reasoning with general queries and without supervision, revolutionizing reasoning scaling laws given the vast availability of general queries. The code will be released at this https URL. 

**Abstract (ZH)**: 提升大规模语言模型推理能力引起了广泛兴趣。然而，当前的后训练技术严重依赖于外部监督信号，如结果监督或辅助奖励模型，这面临着可扩展性和高标注成本的问题。这促使我们无需外部监督来增强语言模型的推理能力。我们提出了一种通用且完全无监督的自训练框架，名为Genius。Genius不要求外部辅助，而是逐步寻找最优的响应序列并优化语言模型。为了探索潜在的步骤并利用最优步骤，Genius引入了一种逐步前瞻采样策略，通过模拟未来结果来采样和估计步长值。此外，我们认识到无监督设置不可避免地会产生内在噪声和不确定性。为提供稳健的优化，我们提出了一种优势校准优化（ACO）损失函数来减轻估计不一致性。结合这些技术，Genius为在通用查询下无监督地自我提升语言模型推理能力奠定了先进基础，革命性地改变了推理的扩展规律，鉴于通用查询的大量可用性。代码将在以下链接发布：https://github.com/Genius-Improving-Reasoning。 

---
# Variability-Driven User-Story Generation using LLM and Triadic Concept Analysis 

**Title (ZH)**: 基于变异性用户的故事情景生成：结合LLM和三元概念分析 

**Authors**: Alexandre Bazin, Alain Gutierrez, Marianne Huchard, Pierre Martin, Yulin, Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08666)  

**Abstract**: A widely used Agile practice for requirements is to produce a set of user stories (also called ``agile product backlog''), which roughly includes a list of pairs (role, feature), where the role handles the feature for a certain purpose. In the context of Software Product Lines, the requirements for a family of similar systems is thus a family of user-story sets, one per system, leading to a 3-dimensional dataset composed of sets of triples (system, role, feature). In this paper, we combine Triadic Concept Analysis (TCA) and Large Language Model (LLM) prompting to suggest the user-story set required to develop a new system relying on the variability logic of an existing system family. This process consists in 1) computing 3-dimensional variability expressed as a set of TCA implications, 2) providing the designer with intelligible design options, 3) capturing the designer's selection of options, 4) proposing a first user-story set corresponding to this selection, 5) consolidating its validity according to the implications identified in step 1, while completing it if necessary, and 6) leveraging LLM to have a more comprehensive website. This process is evaluated with a dataset comprising the user-story sets of 67 similar-purpose websites. 

**Abstract (ZH)**: 一种广泛使用的敏捷实践是生成一组用户故事（也称为“敏捷产品待办事项列表”），这些用户故事大致包括一组角色和功能的配对，其中角色出于某种目的处理该功能。在软件产品线的背景下，对于一组相似系统的功能需求是一个用户故事集的家族，每个系统一个集，构成一个三维数据集，由系统、角色、功能的三元组集合组成。在本文中，我们结合三元概念分析（TCA）和大型语言模型（LLM）提示，根据现有系统家族的变异性逻辑，建议开发新系统所需的用户故事集。该过程包括：1）计算作为一组TCA推论表达的三维变异性；2）为设计者提供可理解的设计选项；3）捕获设计者的选择；4）提出与这些选择相对应的第一个用户故事集；5）根据步骤1中识别的推论验证其有效性，并在必要时进行补充；6）利用LLM获得更全面的网站。本文使用67个相似用途的网站用户故事集数据集对该过程进行了评估。 

---
# Hallucination, reliability, and the role of generative AI in science 

**Title (ZH)**: 幻觉、可靠性和生成性AI在科学中的作用 

**Authors**: Charles Rathkopf  

**Link**: [PDF](https://arxiv.org/pdf/2504.08526)  

**Abstract**: Generative AI is increasingly used in scientific domains, from protein folding to climate modeling. But these models produce distinctive errors known as hallucinations - outputs that are incorrect yet superficially plausible. Worse, some arguments suggest that hallucinations are an inevitable consequence of the mechanisms underlying generative inference. Fortunately, such arguments rely on a conception of hallucination defined solely with respect to internal properties of the model, rather than in reference to the empirical target system. This conception fails to distinguish epistemically benign errors from those that threaten scientific inference. I introduce the concept of corrosive hallucination to capture the epistemically troubling subclass: misrepresentations that are substantively misleading and resistant to systematic anticipation. I argue that although corrosive hallucinations do pose a threat to scientific reliability, they are not inevitable. Scientific workflows such as those surrounding AlphaFold and GenCast, both of which serve as case studies, can neutralize their effects by imposing theoretical constraints during training, and by strategically screening for errors at inference time. When embedded in such workflows, generative AI can reliably contribute to scientific knowledge. 

**Abstract (ZH)**: 生成式AI在科学领域中的应用日益广泛，从蛋白质折叠到气候建模。然而，这些模型会产生一种称为错觉的独特错误——这些输出虽然表面上看似合理，但实际上却是错误的。更糟糕的是，一些论点表明，错觉可能是生成推理机制不可避免的后果。幸运的是，这些论点依赖于仅基于模型内部属性来定义错觉的概念，而不是参照实际的目标系统。这种概念无法区分那些在科学推理中本质上无害的错误和那些构成威胁的错误。我引入了腐蚀性错觉的概念，以捕捉这一类本质上有问题的子类：那些实质上误导且难以系统预见的误导性描述。我认为，尽管腐蚀性错觉确实对科学可靠性构成威胁，但它们并非不可避免。如围绕AlphaFold和GenCast的工作流程，这些可以防范其影响，通过在训练过程中施加理论约束，并在推理时战略性地筛选错误。在这些工作流程中嵌入生成式AI可以确保其对科学知识的可靠贡献。 

---
# Adopting Large Language Models to Automated System Integration 

**Title (ZH)**: 采用大型语言模型进行自动化系统集成 

**Authors**: Robin D. Pesl  

**Link**: [PDF](https://arxiv.org/pdf/2504.08490)  

**Abstract**: Modern enterprise computing systems integrate numerous subsystems to resolve a common task by yielding emergent behavior. A widespread approach is using services implemented with Web technologies like REST or OpenAPI, which offer an interaction mechanism and service documentation standard, respectively. Each service represents a specific business functionality, allowing encapsulation and easier maintenance. Despite the reduced maintenance costs on an individual service level, increased integration complexity arises. Consequently, automated service composition approaches have arisen to mitigate this issue. Nevertheless, these approaches have not achieved high acceptance in practice due to their reliance on complex formal modeling. Within this Ph.D. thesis, we analyze the application of Large Language Models (LLMs) to automatically integrate the services based on a natural language input. The result is a reusable service composition, e.g., as program code. While not always generating entirely correct results, the result can still be helpful by providing integration engineers with a close approximation of a suitable solution, which requires little effort to become operational. Our research involves (i) introducing a software architecture for automated service composition using LLMs, (ii) analyzing Retrieval Augmented Generation (RAG) for service discovery, (iii) proposing a novel natural language query-based benchmark for service discovery, and (iv) extending the benchmark to complete service composition scenarios. We have presented our software architecture as Compositio Prompto, the analysis of RAG for service discovery, and submitted a proposal for the service discovery benchmark. Open topics are primarily the extension of the service discovery benchmark to service composition scenarios and the improvements of the service composition generation, e.g., using fine-tuning or LLM agents. 

**Abstract (ZH)**: 基于大规模语言模型的自动服务集成研究 

---
# Beyond Self-Reports: Multi-Observer Agents for Personality Assessment in Large Language Models 

**Title (ZH)**: 超越自我报告：大型语言模型中的人格评估多观察者代理方法 

**Authors**: Yin Jou Huang, Rafik Hadfi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08399)  

**Abstract**: There is a growing interest in assessing the personality traits of Large language models (LLMs). However, traditional personality assessments based on self-report questionnaires may fail to capture their true behavioral nuances due to inherent biases and meta-knowledge contamination. This paper introduces a novel multi-observer framework for LLM personality assessment that draws inspiration from informant-report methods in psychology. Instead of relying solely on self-assessments, our approach employs multiple observer agents configured with a specific relationship context (e.g., family, friend, or workplace) to simulate interactive scenarios with a subject LLM. These observers engage in dialogues and subsequently provide ratings across the Big Five personality dimensions. Our experiments reveal that LLMs possess systematic biases in self-report personality ratings. Moreover, aggregating observer ratings effectively reduces non-systematic biases and achieves optimal reliability with 5-7 observers. The findings highlight the significant impact of relationship context on personality perception and demonstrate that a multi-observer paradigm yields a more robust and context-sensitive evaluation of LLM personality traits. 

**Abstract (ZH)**: 大型语言模型的人格特质评估：一种多观察者框架的研究 

---
# PCA-RAG: Principal Component Analysis for Efficient Retrieval-Augmented Generation 

**Title (ZH)**: PCA-RAG: 主成分分析辅助的高效检索增强生成 

**Authors**: Arman Khaledian, Amirreza Ghadiridehkordi, Nariman Khaledian  

**Link**: [PDF](https://arxiv.org/pdf/2504.08386)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for grounding large language models in external knowledge sources, improving the precision of agents responses. However, high-dimensional language model embeddings, often in the range of hundreds to thousands of dimensions, can present scalability challenges in terms of storage and latency, especially when processing massive financial text corpora. This paper investigates the use of Principal Component Analysis (PCA) to reduce embedding dimensionality, thereby mitigating computational bottlenecks without incurring large accuracy losses. We experiment with a real-world dataset and compare different similarity and distance metrics under both full-dimensional and PCA-compressed embeddings. Our results show that reducing vectors from 3,072 to 110 dimensions provides a sizeable (up to $60\times$) speedup in retrieval operations and a $\sim 28.6\times$ reduction in index size, with only moderate declines in correlation metrics relative to human-annotated similarity scores. These findings demonstrate that PCA-based compression offers a viable balance between retrieval fidelity and resource efficiency, essential for real-time systems such as Zanista AI's \textit{Newswitch} platform. Ultimately, our study underscores the practicality of leveraging classical dimensionality reduction techniques to scale RAG architectures for knowledge-intensive applications in finance and trading, where speed, memory efficiency, and accuracy must jointly be optimized. 

**Abstract (ZH)**: 基于主成分分析的大规模语言模型嵌入维数缩减在金融文本处理中的应用 

---
# SortBench: Benchmarking LLMs based on their ability to sort lists 

**Title (ZH)**: SortBench: 根据列表排序能力评估LLM模型 

**Authors**: Steffen Herbold  

**Link**: [PDF](https://arxiv.org/pdf/2504.08312)  

**Abstract**: Sorting is a tedious but simple task for human intelligence and can be solved fairly easily algorithmically. However, for Large Language Models (LLMs) this task is surprisingly hard, as some properties of sorting are among known weaknesses of LLMs: being faithful to the input data, logical comparisons between values, and strictly differentiating between syntax (used for sorting) and semantics (typically learned by embeddings). Within this paper, we describe the new SortBench benchmark for LLMs that comes with different difficulties and that can be easily scaled in terms of difficulty. We apply this benchmark to seven state-of-the-art LLMs, including current test-time reasoning models. Our results show that while the o3-mini model is very capable at sorting in general, even this can be fooled if strings are defined to mix syntactical and semantical aspects, e.g., by asking to sort numbers written-out as word. Furthermore, all models have problems with the faithfulness to the input of long lists, i.e., they drop items and add new ones. Our results also show that test-time reasoning has a tendency to overthink problems which leads to performance degradation. Finally, models without test-time reasoning like GPT-4o are not much worse than reasoning models. 

**Abstract (ZH)**: 大语言模型的排序任务：SortBench基准及其挑战 

---
# Large language models could be rote learners 

**Title (ZH)**: 大型语言模型可能是机械记忆的学习者 

**Authors**: Yuyang Xu, Renjun Hu, Haochao Ying, Jian Wu, Xing Shi, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.08300)  

**Abstract**: Multiple-choice question (MCQ) benchmarks are widely used for evaluating Large Language Models (LLMs), yet their reliability is undermined by benchmark contamination. In this study, we reframe contamination as an inherent aspect of learning and seek to disentangle genuine capability acquisition from superficial memorization in LLM evaluation. First, by analyzing model performance under different memorization conditions, we uncover a counterintuitive trend: LLMs perform worse on memorized MCQs than on non-memorized ones, indicating the coexistence of two distinct learning phenomena, i.e., rote memorization and genuine capability learning. To disentangle them, we propose TrinEval, a novel evaluation framework that reformulates MCQs into an alternative trinity format, reducing memorization while preserving knowledge assessment. Experiments validate TrinEval's effectiveness in reformulation, and its evaluation reveals that common LLMs may memorize by rote 20.5% of knowledge points (in MMLU on average). 

**Abstract (ZH)**: Multiple-choice question benchmarks的可靠性受到基准污染的威胁：一种重新审视大型语言模型评估的新框架 

---
# ELSA: A Style Aligned Dataset for Emotionally Intelligent Language Generation 

**Title (ZH)**: ELSA：一种情感对齐的数据集，用于情感智能语言生成 

**Authors**: Vishal Gandhi, Sagar Gandhi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08281)  

**Abstract**: Advancements in emotion aware language processing increasingly shape vital NLP applications ranging from conversational AI and affective computing to computational psychology and creative content generation. Existing emotion datasets either lack emotional granularity or fail to capture necessary stylistic diversity, limiting the advancement of effective emotion conditioned text generation systems. Seeking to bridge this crucial gap between granularity and style diversity, this paper introduces a novel systematically constructed dataset named ELSA Emotion and Language Style Alignment Dataset leveraging fine grained emotion taxonomies adapted from existing sources such as dair ai emotion dataset and GoEmotions taxonomy. This dataset comprises multiple emotionally nuanced variations of original sentences regenerated across distinct contextual styles such as conversational, formal, poetic, and narrative, using advanced Large Language Models LLMs. Rigorous computational evaluation using metrics such as perplexity, embedding variance, readability, lexical diversity, and semantic coherence measures validates the datasets emotional authenticity, linguistic fluency, and textual diversity. Comprehensive metric analyses affirm its potential to support deeper explorations into emotion conditioned style adaptive text generation. By enabling precision tuned emotionally nuanced language modeling, our dataset creates fertile ground for research on fine grained emotional control, prompt driven explanation, interpretability, and style adaptive expressive language generation with LLMs. 

**Abstract (ZH)**: 情感感知语言处理的进展日益塑造了从对话AI和情感计算到计算心理学和创意内容生成等关键的NLP应用。现有的情感数据集要么缺乏情感细腻度，要么未能捕捉必要的风格多样性，限制了有效的情感条件文本生成系统的发展。为弥合细腻度与风格多样性之间的关键差距，本文介绍了一种名为ELSA情感与语言风格对齐数据集的新颖系统构建数据集，该数据集利用从现有来源（如dair ai情感数据集和GoEmotions分类法）改编的精细情感分类法。该数据集包含使用先进大型语言模型（LLMs）跨越不同语境风格（如对话、正式、诗歌和叙事）重新生成的多种情感细微差异的原始句子变体。使用诸如困惑度、嵌入变化、可读性、词汇多样性以及语义一致性等严格计算评估指标，验证了数据集的情感真实性、语言流畅性和文本多样性。全面的度量分析证实其支持对情感条件下的风格适应性文本生成更深入探索的潜力。通过实现精准调优的情感细腻语言建模，我们的数据集为基于LLMs的情感细粒度控制、提示驱动解释、可解释性和风格适应性表达语言生成的研究奠定了基础。 

---
# RAG-VR: Leveraging Retrieval-Augmented Generation for 3D Question Answering in VR Environments 

**Title (ZH)**: RAG-VR：利用检索增强生成技术进行VR环境中的3D问答 

**Authors**: Shiyi Ding, Ying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.08256)  

**Abstract**: Recent advances in large language models (LLMs) provide new opportunities for context understanding in virtual reality (VR). However, VR contexts are often highly localized and personalized, limiting the effectiveness of general-purpose LLMs. To address this challenge, we present RAG-VR, the first 3D question-answering system for VR that incorporates retrieval-augmented generation (RAG), which augments an LLM with external knowledge retrieved from a localized knowledge database to improve the answer quality. RAG-VR includes a pipeline for extracting comprehensive knowledge about virtual environments and user conditions for accurate answer generation. To ensure efficient retrieval, RAG-VR offloads the retrieval process to a nearby edge server and uses only essential information during retrieval. Moreover, we train the retriever to effectively distinguish among relevant, irrelevant, and hard-to-differentiate information in relation to questions. RAG-VR improves answer accuracy by 17.9%-41.8% and reduces end-to-end latency by 34.5%-47.3% compared with two baseline systems. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）为虚拟现实（VR）中的上下文理解提供了新的机遇。然而，VR上下文往往高度局部化和个性化，限制了通用大型语言模型的效果。为应对这一挑战，我们提出了RAG-VR，这是第一个结合检索增强生成（RAG）技术的3D问答系统，该系统通过从局部知识数据库中检索外部知识来增强大型语言模型，以提高答案质量。RAG-VR包括一个提取虚拟环境和用户条件全面知识的管道，以实现准确的回答生成。为了确保高效的检索，RAG-VR将检索过程卸载到附近的边缘服务器，并仅在检索过程中使用必要的信息。此外，我们训练检索器有效地区分与问题相关的、无关的以及难以区分的信息。与两个基线系统相比，RAG-VR将答案准确性提高了17.9%-41.8%，并将端到端延迟降低了34.5%-47.3%。 

---
# Jupiter: Fast and Resource-Efficient Collaborative Inference of Generative LLMs on Edge Devices 

**Title (ZH)**: Jupiter：在边缘设备上快速且资源高效的生成型LLM协作推理 

**Authors**: Shengyuan Ye, Bei Ouyang, Liekang Zeng, Tianyi Qian, Xiaowen Chu, Jian Tang, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.08242)  

**Abstract**: Generative large language models (LLMs) have garnered significant attention due to their exceptional capabilities in various AI tasks. Traditionally deployed in cloud datacenters, LLMs are now increasingly moving towards more accessible edge platforms to protect sensitive user data and ensure privacy preservation. The limited computational resources of individual edge devices, however, can result in excessively prolonged inference latency and overwhelmed memory usage. While existing research has explored collaborative edge computing to break the resource wall of individual devices, these solutions yet suffer from massive communication overhead and under-utilization of edge resources. Furthermore, they focus exclusively on optimizing the prefill phase, neglecting the crucial autoregressive decoding phase for generative LLMs. To address that, we propose Jupiter, a fast, scalable, and resource-efficient collaborative edge AI system for generative LLM inference. Jupiter introduces a flexible pipelined architecture as a principle and differentiates its system design according to the differentiated characteristics of the prefill and decoding phases. For prefill phase, Jupiter submits a novel intra-sequence pipeline parallelism and develops a meticulous parallelism planning strategy to maximize resource efficiency; For decoding, Jupiter devises an effective outline-based pipeline parallel decoding mechanism combined with speculative decoding, which further magnifies inference acceleration. Extensive evaluation based on realistic implementation demonstrates that Jupiter remarkably outperforms state-of-the-art approaches under various edge environment setups, achieving up to 26.1x end-to-end latency reduction while rendering on-par generation quality. 

**Abstract (ZH)**: 生成型大型语言模型（LLMs）因其在各种AI任务中的出色能力而备受关注。传统上部署在云数据中心的LLMs现在越来越多地转向更易于访问的边缘平台，以保护敏感用户数据和确保隐私保护。然而，个体边缘设备有限的计算资源可能导致推理延迟过长和内存使用过载。尽管现有研究探索了协作边缘计算以突破个体设备的资源壁垒，但这些解决方案仍然面临巨大的通信开销和边缘资源利用率低的问题。此外，它们仅专注于优化预填阶段，忽视了生成型LLMs至关重要的自回归解码阶段。为此，我们提出了一种名为Jupiter的快速、可扩展且资源高效的协作边缘AI系统，用于生成型LLM推理。Jupiter引入了灵活的流水线架构作为其设计原则，并根据预填阶段和解码阶段的不同特征差异化其系统设计。对于预填阶段，Jupiter提交了一种新颖的序列内流水线并行性，并开发了精细的并行性规划策略以最大化资源效率；对于解码，Jupiter设计了一种有效的基于大纲的流水线并行解码机制结合推测性解码，从而进一步提升推理加速效果。基于现实部署的广泛评估表明，在各种边缘环境配置下，Jupiter显著优于现有最佳方法，实现端到端延迟最多减少26.1倍，同时保持生成质量相当。 

---
# LLM for Comparative Narrative Analysis 

**Title (ZH)**: 大规模语言模型在叙事比较分析中的应用 

**Authors**: Leo Kampen, Carlos Rabat Villarreal, Louis Yu, Santu Karmaker, Dongji Feng  

**Link**: [PDF](https://arxiv.org/pdf/2504.08211)  

**Abstract**: In this paper, we conducted a Multi-Perspective Comparative Narrative Analysis (CNA) on three prominent LLMs: GPT-3.5, PaLM2, and Llama2. We applied identical prompts and evaluated their outputs on specific tasks, ensuring an equitable and unbiased comparison between various LLMs. Our study revealed that the three LLMs generated divergent responses to the same prompt, indicating notable discrepancies in their ability to comprehend and analyze the given task. Human evaluation was used as the gold standard, evaluating four perspectives to analyze differences in LLM performance. 

**Abstract (ZH)**: 本文对三款 prominant LLM（GPT-3.5, PaLM2, 和 Llama2）进行了多视角比较叙事分析（CNA），采用了相同的提示并评估了它们在特定任务上的输出，确保了对各种LLM进行公平和无偏见的比较。研究结果显示，三款LLM对同一提示的响应存在显著差异，表明它们在理解和分析给定任务方面的能力存在明显差异。人类评估被用作黄金标准，从四个视角分析了LLM性能的差异。 

---
# How Good Are Large Language Models for Course Recommendation in MOOCs? 

**Title (ZH)**: 大型语言模型在MOOC课程推荐中的能力如何？ 

**Authors**: Boxuan Ma, Md Akib Zabed Khan, Tianyuan Yang, Agoritsa Polyzou, Shin'ichi Konomi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08208)  

**Abstract**: Large Language Models (LLMs) have made significant strides in natural language processing and are increasingly being integrated into recommendation systems. However, their potential in educational recommendation systems has yet to be fully explored. This paper investigates the use of LLMs as a general-purpose recommendation model, leveraging their vast knowledge derived from large-scale corpora for course recommendation tasks. We explore a variety of approaches, ranging from prompt-based methods to more advanced fine-tuning techniques, and compare their performance against traditional recommendation models. Extensive experiments were conducted on a real-world MOOC dataset, evaluating using LLMs as course recommendation systems across key dimensions such as accuracy, diversity, and novelty. Our results demonstrate that LLMs can achieve good performance comparable to traditional models, highlighting their potential to enhance educational recommendation systems. These findings pave the way for further exploration and development of LLM-based approaches in the context of educational recommendations. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言处理领域取得了显著进展，并越来越多地被集成到推荐系统中。然而，它们在教育推荐系统中的潜力尚未得到充分探索。本文研究了将LLMs作为通用推荐模型的应用，利用其从大规模语料库中获得的广泛知识，进行课程推荐任务。我们探索了从提示方法到更高级的微调技术等多种方法，并将它们的性能与传统推荐模型进行了比较。我们在一个真实的MOOC数据集上进行了广泛的实验，从准确度、多样性和新颖性等多个维度评估了使用LLMs作为课程推荐系统的表现。我们的研究结果表明，LLMs可以实现与传统模型相当的性能，突显了其增强教育推荐系统潜力的可能性。这些发现为探索和开发基于LLM的方法在教育推荐领域的应用铺平了道路。 

---
# DRAFT-ing Architectural Design Decisions using LLMs 

**Title (ZH)**: 使用大型语言模型草图化架构设计决策 

**Authors**: Rudra Dhar, Adyansh Kakran, Amey Karan, Karthik Vaidhyanathan, Vasudeva Varma  

**Link**: [PDF](https://arxiv.org/pdf/2504.08207)  

**Abstract**: Architectural Knowledge Management (AKM) is crucial for software development but remains challenging due to the lack of standardization and high manual effort. Architecture Decision Records (ADRs) provide a structured approach to capture Architecture Design Decisions (ADDs), but their adoption is limited due to the manual effort involved and insufficient tool support. Our previous work has shown that Large Language Models (LLMs) can assist in generating ADDs. However, simply prompting the LLM does not produce quality ADDs. Moreover, using third-party LLMs raises privacy concerns, while self-hosting them poses resource challenges.
To this end, we experimented with different approaches like few-shot, retrieval-augmented generation (RAG) and fine-tuning to enhance LLM's ability to generate ADDs. Our results show that both techniques improve effectiveness. Building on this, we propose Domain Specific Retreival Augumented Few Shot Fine Tuninng, DRAFT, which combines the strengths of all these three approaches for more effective ADD generation. DRAFT operates in two phases: an offline phase that fine-tunes an LLM on generating ADDs augmented with retrieved examples and an online phase that generates ADDs by leveraging retrieved ADRs and the fine-tuned model.
We evaluated DRAFT against existing approaches on a dataset of 4,911 ADRs and various LLMs and analyzed them using automated metrics and human evaluations. Results show DRAFT outperforms all other approaches in effectiveness while maintaining efficiency. Our findings indicate that DRAFT can aid architects in drafting ADDs while addressing privacy and resource constraints. 

**Abstract (ZH)**: 特定领域检索增强少量样本细调方法（DRAFT）：提高架构设计决策生成的有效性和效率 

---
# SAEs $\textit{Can}$ Improve Unlearning: Dynamic Sparse Autoencoder Guardrails for Precision Unlearning in LLMs 

**Title (ZH)**: SAEs 可以提升未学习能力：动态稀疏自编码器在大规模语言模型中精确未学习的界限 

**Authors**: Aashiq Muhamed, Jacopo Bonato, Mona Diab, Virginia Smith  

**Link**: [PDF](https://arxiv.org/pdf/2504.08192)  

**Abstract**: Machine unlearning is a promising approach to improve LLM safety by removing unwanted knowledge from the model. However, prevailing gradient-based unlearning methods suffer from issues such as high computational costs, hyperparameter instability, poor sequential unlearning capability, vulnerability to relearning attacks, low data efficiency, and lack of interpretability. While Sparse Autoencoders are well-suited to improve these aspects by enabling targeted activation-based unlearning, prior approaches underperform gradient-based methods. This work demonstrates that, contrary to these earlier findings, SAEs can significantly improve unlearning when employed dynamically. We introduce $\textbf{Dynamic DAE Guardrails}$ (DSG), a novel method for precision unlearning that leverages principled feature selection and a dynamic classifier. Our experiments show DSG substantially outperforms leading unlearning methods, achieving superior forget-utility trade-offs. DSG addresses key drawbacks of gradient-based approaches for unlearning -- offering enhanced computational efficiency and stability, robust performance in sequential unlearning, stronger resistance to relearning attacks, better data efficiency including zero-shot settings, and more interpretable unlearning. 

**Abstract (ZH)**: 动态稀疏自编码器-guardrails（DSG）：一种精确遗忘的新方法 

---
# Geneshift: Impact of different scenario shift on Jailbreaking LLM 

**Title (ZH)**: Geneshift: 不同场景转变对破解LLM的影响 

**Authors**: Tianyi Wu, Zhiwei Xue, Yue Liu, Jiaheng Zhang, Bryan Hooi, See-Kiong Ng  

**Link**: [PDF](https://arxiv.org/pdf/2504.08104)  

**Abstract**: Jailbreak attacks, which aim to cause LLMs to perform unrestricted behaviors, have become a critical and challenging direction in AI safety. Despite achieving the promising attack success rate using dictionary-based evaluation, existing jailbreak attack methods fail to output detailed contents to satisfy the harmful request, leading to poor performance on GPT-based evaluation. To this end, we propose a black-box jailbreak attack termed GeneShift, by using a genetic algorithm to optimize the scenario shifts. Firstly, we observe that the malicious queries perform optimally under different scenario shifts. Based on it, we develop a genetic algorithm to evolve and select the hybrid of scenario shifts. It guides our method to elicit detailed and actionable harmful responses while keeping the seemingly benign facade, improving stealthiness. Extensive experiments demonstrate the superiority of GeneShift. Notably, GeneShift increases the jailbreak success rate from 0% to 60% when direct prompting alone would fail. 

**Abstract (ZH)**: 基因突变：一种基于遗传算法的黑盒 Jailbreak 攻击 

---
# Can Reasoning LLMs Enhance Clinical Document Classification? 

**Title (ZH)**: 基于推理的大型语言模型能否增强临床文档分类？ 

**Authors**: Akram Mustafa, Usman Naseem, Mostafa Rahimi Azghadi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08040)  

**Abstract**: Clinical document classification is essential for converting unstructured medical texts into standardised ICD-10 diagnoses, yet it faces challenges due to complex medical language, privacy constraints, and limited annotated datasets. Large Language Models (LLMs) offer promising improvements in accuracy and efficiency for this task. This study evaluates the performance and consistency of eight LLMs; four reasoning (Qwen QWQ, Deepseek Reasoner, GPT o3 Mini, Gemini 2.0 Flash Thinking) and four non-reasoning (Llama 3.3, GPT 4o Mini, Gemini 2.0 Flash, Deepseek Chat); in classifying clinical discharge summaries using the MIMIC-IV dataset. Using cTAKES to structure clinical narratives, models were assessed across three experimental runs, with majority voting determining final predictions. Results showed that reasoning models outperformed non-reasoning models in accuracy (71% vs 68%) and F1 score (67% vs 60%), with Gemini 2.0 Flash Thinking achieving the highest accuracy (75%) and F1 score (76%). However, non-reasoning models demonstrated greater stability (91% vs 84% consistency). Performance varied across ICD-10 codes, with reasoning models excelling in complex cases but struggling with abstract categories. Findings indicate a trade-off between accuracy and consistency, suggesting that a hybrid approach could optimise clinical coding. Future research should explore multi-label classification, domain-specific fine-tuning, and ensemble methods to enhance model reliability in real-world applications. 

**Abstract (ZH)**: 临床文档分类对于将未结构化医疗文本转换为标准化ICD-10诊断至关重要，但由于医学语言复杂、隐私限制以及标注数据集有限，该任务面临挑战。大型语言模型（LLMs）在提高准确性和效率方面展现出 promising 的潜力。本研究评估了八种LLMs（四种推理模型和四种非推理模型）在使用MIMIC-IV数据集分类临床出院总结方面的性能和一致性。通过cTAKES结构化临床叙事，模型在三次实验运行中进行了评估，采用多数投票确定最终预测结果。结果显示，推理模型在准确率（71% vs 68%）和F1分数（67% vs 60%）方面优于非推理模型，其中Gemini 2.0 Flash Thinking获得最高准确率（75%）和F1分数（76%）。然而，非推理模型在稳定性和一致性方面表现更好（91% vs 84%）。不同ICD-10代码的表现有所差异，推理模型在复杂案例中表现优异但在抽象类别方面遇到困难。研究结果表明，准确性和一致性之间存在权衡，这表明混合方法可能优化临床编码。未来研究应探索多标签分类、领域特异性微调和集成方法以提高模型在实际应用中的可靠性。 

---
# 'Neural howlround' in large language models: a self-reinforcing bias phenomenon, and a dynamic attenuation solution 

**Title (ZH)**: 大型语言模型中的“神经共鸣”：一种自我增强偏差现象及动态衰减解决方案 

**Authors**: Seth Drake  

**Link**: [PDF](https://arxiv.org/pdf/2504.07992)  

**Abstract**: Large language model (LLM)-driven AI systems may exhibit an inference failure mode we term `neural howlround,' a self-reinforcing cognitive loop where certain highly weighted inputs become dominant, leading to entrenched response patterns resistant to correction. This paper explores the mechanisms underlying this phenomenon, which is distinct from model collapse and biased salience weighting. We propose an attenuation-based correction mechanism that dynamically introduces counterbalancing adjustments and can restore adaptive reasoning, even in `locked-in' AI systems. Additionally, we discuss some other related effects arising from improperly managed reinforcement. Finally, we outline potential applications of this mitigation strategy for improving AI robustness in real-world decision-making tasks. 

**Abstract (ZH)**: 大型语言模型驱动的AI系统可能存在一种我们称为“神经回响”的推理失败模式，这是一种自强化的认知循环，其中某些高权重输入变得主导，导致顽固的响应模式难以纠正。本文探讨了这一现象的机理，该现象不同于模型崩溃和偏差显著性加权。我们提出了一种基于衰减的纠正机制，能够动态引入平衡调整，即使在“锁定”AI系统中也能恢复适应性推理。此外，我们还讨论了由于不当强化管理而产生的一些相关效应。最后，我们概述了这一缓解策略在提高实际决策任务中AI鲁棒性方面的潜在应用。 

---
# Regional Tiny Stories: Using Small Models to Compare Language Learning and Tokenizer Performance 

**Title (ZH)**: 区域微小故事：使用小型模型比较语言学习和分词器性能 

**Authors**: Nirvan Patil, Malhar Abhay Inamdar, Agnivo Gosai, Guruprasad Pathak, Anish Joshi, Aryan Sagavekar, Anish Joshirao, Raj Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2504.07989)  

**Abstract**: Small Language Models (SLMs) offer efficient alternatives to LLMs for specific domains. The 2023 TinyStories study developed an English dataset that allows SLMs with 1 to 10 million parameters to produce coherent outputs. Our research expands this framework by translating the original dataset into Indian languages and creating synthetic data using LLMs. We focus on Hindi, Marathi, and Bengali, evaluating SLMs for regional language processing and understanding linguistic complexity. We show that SLMs efficiently process regional languages with significantly fewer parameters than LLMs, providing a complementary framework for ``inference based evaluation" of tokenization strategies and linguistic complexity. Our analysis shows that language-specific tokenizers outperform general-purpose ones for Indian languages. Empirical validations, supported by information-theoretic and morphological analyses, provides fundamental understanding behind the better performance of Hindi models over Marathi and Bengali. Additionally, we show that synthetic datasets outperform translated content for training SLMs. Correlation analyses reveal cross-linguistic patterns and language-specific relationships between creativity, grammatical precision, and narrative completeness. These findings advance both the practical application of SLMs to underserved languages and our theoretical understanding of neural language development. 

**Abstract (ZH)**: 小型语言模型（SLMs）为特定领域提供了LLMs的高效替代方案。2023年TinyStories研究开发了一个英语数据集，使参数量在1至1000万之间的SLMs能够生成连贯的输出。我们的研究扩展了这一框架，将原始数据集翻译成印度语言，并使用LLMs生成合成数据。我们专注于北部印地语、马拉地语和孟加拉语，评估SLMs在区域语言处理中的表现，以及理解语言复杂性。我们展示了SLMs能够用远少于LLMs的参数高效处理区域语言，为基于推断的词元化策略和语言复杂性评估提供了补充框架。我们的分析表明，特定于语言的词元化器在印度语处理上优于通用词元化器。信息论和形态学分析的支持下的实证验证，揭示了印地语模型在马拉地语和孟加拉语中表现更好的基本原理。此外，我们展示了合成数据集在训练SLMs时优于翻译内容。相关性分析揭示了跨语言模式和语言特异性关系，这些关系涉及创造力、语法精确性和叙事完整性。这些发现不仅推进了对未服务语言中SLMs应用的实际意义，还加深了我们对神经语言发展理论的理解。 

---
# SEAL: Steerable Reasoning Calibration of Large Language Models for Free 

**Title (ZH)**: SEAL: 可引导的大型语言模型推理校准以实现自由推理 

**Authors**: Runjin Chen, Zhenyu Zhang, Junyuan Hong, Souvik Kundu, Zhangyang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.07986)  

**Abstract**: Large Language Models (LLMs), such as OpenAI's o1-series have demonstrated compelling capabilities for complex reasoning tasks via the extended chain-of-thought (CoT) reasoning mechanism. However, recent studies reveal substantial redundancy in the CoT reasoning traces, which not only increases inference latency but also negatively impacts model performance by diverting attention to unnecessary reasoning paths. To address this issue, we investigate the internal reasoning structures of LLMs and categorize them into three primary thought types: execution, reflection, and transition thoughts. Moreover, our analysis reveals that excessive reflection and transition thoughts are strongly correlated with failure cases and these thought categories exhibit clear separation in the latent space. Based on these, we introduce SEAL (Steerable reasoning calibration), a training-free approach that seamlessly calibrates the CoT process, improving accuracy while demonstrating significant efficiency gains. SEAL consists of an offline stage for extracting the reasoning steering vector in the latent space, followed by an on-the-fly calibration of the reasoning trace through representation intervention using the steering vector. Notably, the steering vector exhibits strong transferability across various tasks. Extensive experiments across multiple models (DeepSeek-R1-Distill and QwQ-32B-Preview) and benchmarks (Math500, GSM8K, LiveCodeBench) validate the effectiveness of SEAL, up to a 11% improvement in accuracy while reducing reasoning tokens by 11.8% to 50.4%. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）中的可导航推理校准：大型语言模型，如OpenAI的o1系列，通过扩展的链式思考（CoT）推理机制展示了强大的复杂推理能力。然而，最近的研究揭示了CoT推理轨迹中的大量冗余，这不仅增加了推理延迟，还通过分散注意力到不必要的推理路径上而负面影响了模型性能。为了解决这一问题，我们研究了LLMs的内部推理结构，并将其分类为三种主要的思维类型：执行思维、反思思维和过渡思维。此外，我们的分析表明，过度的反思思维和过渡思维与失败案例密切相关，这些思维类别在潜在空间中表现出明确的分离。基于这些发现，我们引入了SEAL（可导航的推理校准），这是一种无需训练的方法，能够无缝校准CoT过程，提高准确率同时显著提高效率。SEAL包括一个离线阶段，用于在潜在空间中提取推理导向向量，接着是通过使用导向向量进行表示干预来实时校准推理轨迹。值得注意的是，导向向量在各种任务之间表现出很强的可转移性。在多种模型（DeepSeek-R1-Distill和QwQ-32B-Preview）和基准（Math500、GSM8K、LiveCodeBench）上进行的大量实验证实了SEAL的有效性，准确率最多可提高11%，同时减少推理令牌11.8%至50.4%。我们的代码已公开可在以下网址获取。 

---
# Psychological Health Knowledge-Enhanced LLM-based Social Network Crisis Intervention Text Transfer Recognition Method 

**Title (ZH)**: 基于心理卫生知识增强的LLM社交网络危机干预文本转移识别方法 

**Authors**: Shurui Wu, Xinyi Huang, Dingxin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2504.07983)  

**Abstract**: As the prevalence of mental health crises increases on social media platforms, identifying and preventing potential harm has become an urgent challenge. This study introduces a large language model (LLM)-based text transfer recognition method for social network crisis intervention, enhanced with domain-specific mental health knowledge. We propose a multi-level framework that incorporates transfer learning using BERT, and integrates mental health knowledge, sentiment analysis, and behavior prediction techniques. The framework includes a crisis annotation tool trained on social media datasets from real-world events, enabling the model to detect nuanced emotional cues and identify psychological crises. Experimental results show that the proposed method outperforms traditional models in crisis detection accuracy and exhibits greater sensitivity to subtle emotional and contextual variations. 

**Abstract (ZH)**: 社交媒体平台上心理健康危机频发背景下基于大语言模型的文本转移识别方法及其在危机干预中的应用：融合领域特定心理健康知识的多层次框架 

---
# Metamorphic Testing for Fairness Evaluation in Large Language Models: Identifying Intersectional Bias in LLaMA and GPT 

**Title (ZH)**: 大语言模型中公平性评估的变形测试：识别LLaMA和GPT的交叉偏见 

**Authors**: Harishwar Reddy, Madhusudan Srinivasan, Upulee Kanewala  

**Link**: [PDF](https://arxiv.org/pdf/2504.07982)  

**Abstract**: Large Language Models (LLMs) have made significant strides in Natural Language Processing but remain vulnerable to fairness-related issues, often reflecting biases inherent in their training data. These biases pose risks, particularly when LLMs are deployed in sensitive areas such as healthcare, finance, and law. This paper introduces a metamorphic testing approach to systematically identify fairness bugs in LLMs. We define and apply a set of fairness-oriented metamorphic relations (MRs) to assess the LLaMA and GPT model, a state-of-the-art LLM, across diverse demographic inputs. Our methodology includes generating source and follow-up test cases for each MR and analyzing model responses for fairness violations. The results demonstrate the effectiveness of MT in exposing bias patterns, especially in relation to tone and sentiment, and highlight specific intersections of sensitive attributes that frequently reveal fairness faults. This research improves fairness testing in LLMs, providing a structured approach to detect and mitigate biases and improve model robustness in fairness-sensitive applications. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自然语言处理领域取得了显著进展，但仍存在与公平性相关的问题，往往反映了其训练数据中存在的偏见。这些偏见在LLMs部署于医疗、金融和法律等敏感领域时会带来风险。本文介绍了一种 metamorphic 测试方法，以系统地识别LLMs中的公平性漏洞。我们定义并应用于评估最先进的LLM——LLaMA和GPT模型的一组公平性导向的 metamorphic 关系（MRs），涉及多样化的 demographic 输入。该方法包括为每个MR生成源测试用例和后续测试用例，并分析模型响应以检测公平性违规。结果表明，MT在揭示与语气和情感相关偏见模式方面特别有效，并突显了敏感属性的特定交叉点，这些交叉点经常揭示公平性故障。该研究改善了LLMs的公平性测试，提供了一种结构化的检测和缓解偏见的方法，以提高公平性敏感应用中模型的鲁棒性。 

---
